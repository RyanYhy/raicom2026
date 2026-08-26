#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# raicom2026/scan_logic.py —— 云台扫描去重/瞄准 纯逻辑
# ------------------------------------------------------------
# webcam 云台"识别+打靶"一体化扫描的核心数学/聚类, 与 ROS/硬件完全解耦,
# 可离线单测 (tests/test_scan_logic.py)。
#   1) px→度 (绝对云台角):
#       abs_pan  = pan_cmd  + s_h * (u - W/2) / ppd_h
#       abs_tilt = tilt_cmd + s_v * (v - H/2) / ppd_v
#   2) 聚类去重 (与方案 §3.1 一致):
#       同类别 |Δpan|<5° 且 |Δtilt|<8° → 同一目标
#       tilt 两档重复: 同类别 |Δpan|<5° 且 |Δtilt|<3° → 合并
#       类别冲突: 6° pan 内异类 → 取高置信 (宁丢 +10 不赌 -20)
#   3) 瞄准/开火目标选择 (enemy 安全门 + 闭环微调)
# 类别索引必须与 person.onnx 训练一致: {0:hostage, 1:enemy, 2:friend}
# ============================================================

# ---- person.onnx 类别 ----
HOSTAGE = 0
ENEMY = 1
FRIEND = 2
CLASS_NAMES = {HOSTAGE: 'hostage', ENEMY: 'enemy', FRIEND: 'friend'}

# ---- 聚类容差 (度) ----
CLUSTER_PAN_TOL = 5.0       # 同人水平容差
CLUSTER_TILT_TOL = 8.0      # 同人垂直容差
TILT_REPEAT_TOL = 3.0       # 两档俯仰重复合并容差
CONFLICT_PAN_TOL = 6.0      # 类别冲突判定: 6° pan 内异类取高置信
CONFLICT_TILT_TOL = 10.0    # 类别冲突还需 |Δtilt|<=10° (同 pan 不同高度的真目标不算冲突)
CLUSTER_MIN_CONF = 0.30     # 聚类最低接受置信(低于此不参与计数/打靶)
# 跨档像素连续性: 两档俯仰之间云台转动 |tilt_step|, 同人像素 v 的位移大小
# 必为 |tilt_step|*ppd_v (方向无关!)。用于 s_v 符号未标定时仍不重复计数。
CROSS_LEVEL_PIX_TOL_DEG = 3.0   # 像素位移容差(折成度) -> ~±36px @ ppd=12


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def px_to_pan(pan_cmd, u, img_w, ppd_h, s_h=1.0):
    """像素 u → 目标绝对水平角(度)。s_h 为方向符号因子(标定)。"""
    return pan_cmd + s_h * (u - img_w / 2.0) / ppd_h


def px_to_tilt(tilt_cmd, v, img_h, ppd_v, s_v=1.0):
    """像素 v → 目标绝对俯仰角(度)。s_v 为方向符号因子(标定)。"""
    return tilt_cmd + s_v * (v - img_h / 2.0) / ppd_v


def distance_from_box_height(target_h_m, focal_px, box_h_px):
    """由框高反推距离: d = 目标物理高度 × 焦距 / 框高px (方案 §3.2)。"""
    if box_h_px <= 0 or focal_px <= 0:
        return None
    return target_h_m * focal_px / box_h_px


def tilt_offset(dist, a0=0.0, a1=0.0):
    """水弹道垂直补偿 (度): TILT_OFFSET(d) = a0 + a1/d (方案 §3.2, 标定后填)。
    未标定返回 a0(=0), 即不补偿。"""
    if dist and dist > 0:
        return a0 + a1 / dist
    return a0


class Det:
    """一次扫描中的单个检测(已映射到绝对云台角)。frame/boxes 用于存证据图。
    level 记录检测来自哪一档俯仰命令(用于跨档像素连续性去重, 见 cluster)。"""
    __slots__ = ('cls', 'conf', 'u', 'v', 'w', 'h', 'pan', 'tilt', 'level',
                 'frame', 'boxes', 'scores', 'ids')

    def __init__(self, cls, conf, u, v, w, h, pan, tilt, level=None,
                 frame=None, boxes=None, scores=None, ids=None):
        self.cls, self.conf = cls, conf
        self.u, self.v = u, v
        self.w, self.h = w, h
        self.pan, self.tilt = pan, tilt
        self.level = level                       # 该检测的俯仰命令角(档位)
        self.frame = frame                       # 证据图: 该检测所在帧
        self.boxes = boxes                       # 该帧全部检测(画证据图)
        self.scores = scores
        self.ids = ids


class Cluster:
    """聚成一类的人/物; 中心角度取最高置信检测。"""
    __slots__ = ('cls', 'conf', 'pan', 'tilt', 'best')

    def __init__(self, det):
        self.cls = det.cls
        self.conf = det.conf
        self.pan = det.pan
        self.tilt = det.tilt
        self.best = det

    def __repr__(self):
        return 'Cluster(%s conf=%.2f pan=%.1f tilt=%.1f)' % (
            CLASS_NAMES.get(self.cls, self.cls), self.conf, self.pan, self.tilt)


def _same_person(pan_a, tilt_a, det_a, pan_b, tilt_b, det_b,
                 tilt_step=10.0, ppd_v=12.0, pix_tol_deg=CROSS_LEVEL_PIX_TOL_DEG):
    """跨档(两档俯仰)判定 a/b 是否为同一人, 对 s_v 符号方向无关。
    - 若两档绝对角本就一致 (s_v 正确) -> 常规容差合并
    - 若 s_v 符号相反 (未标定), 同人在两档的绝对 tilt 会差 ±2*tilt_step,
      但其像素 v 的位移大小必为 |tilt_step|*ppd_v —— 用像素连续性判定, 不依赖符号。
    - 仅对"不同档"的检测启用像素规则 (同档两个差 10° 像素的人是两个真目标)。
    """
    if abs(pan_a - pan_b) > CLUSTER_PAN_TOL:
        return False                       # 水平离太远, 必不是同一人
    if abs(tilt_a - tilt_b) <= CLUSTER_TILT_TOL:
        return True                        # 绝对角已一致 -> 常规合并
    # 绝对角不一致: 可能是 s_v 符号未标定导致跨档差 ±2*tilt_step。
    # 用像素连续性判定 (方向无关): 同人在两档的像素 v 位移大小必为 |tilt_step|*ppd_v。
    if det_a is None or det_b is None or det_a.level is None or det_b.level is None:
        return False
    level_gap = abs(det_a.level - det_b.level)
    if abs(level_gap - tilt_step) > 0.5:
        return False                       # 非相邻档, 像素规则不适用
    pix_shift = abs(det_a.v - det_b.v)
    expected = tilt_step * ppd_v
    return abs(pix_shift - expected) <= pix_tol_deg * ppd_v


def cluster_person_detections(dets, tilt_step=10.0, ppd_v=12.0,
                              pix_tol_deg=CROSS_LEVEL_PIX_TOL_DEG):
    """把一次扫描收集的所有检测去重聚类 → Cluster 列表 (按类别再按置信度排序)。
    与方案 §3.1 一致的判定:
      - 同类别 |Δpan|<5° 且 |Δtilt|<8° → 同一目标 (合并, 中心取最高置信检测)
      - 跨档像素连续性: 同类别 pan 相近 + 像素位移 ≈ |tilt_step|*ppd_v → 同一目标
        (s_v 符号未标定时两档绝对角会差 ±2*tilt_step, 常规容差合并不上, 靠这条兜底)
      - tilt 两档重复: 同类别且 |Δpan|<5° 且 |Δtilt|<3° → 再合并
      - 类别冲突: pan <6° 且 |Δtilt|<=10° 内异类 → 保留置信最高者
        (同 pan 但高度差大的两个真目标不算冲突, 避免少报)
    """
    out = []
    for d in sorted(dets, key=lambda x: -x.conf):
        if d.conf < CLUSTER_MIN_CONF:
            continue
        hit = None
        for c in out:
            if c.cls != d.cls:
                continue
            if _same_person(c.pan, c.tilt, c.best, d.pan, d.tilt, d,
                            tilt_step, ppd_v, pix_tol_deg):
                hit = c
                break
        if hit is None:
            out.append(Cluster(d))
        elif d.conf > hit.conf:                  # 保留最高置信检测作中心
            hit.conf = d.conf
            hit.pan = d.pan
            hit.tilt = d.tilt
            hit.best = d

    # ---- tilt 两档重复合并 (同类别, 更紧的俯仰容差) ----
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(out):
            j = i + 1
            while j < len(out):
                a, b = out[i], out[j]
                if (a.cls == b.cls and
                        abs(a.pan - b.pan) <= CLUSTER_PAN_TOL and
                        abs(a.tilt - b.tilt) <= TILT_REPEAT_TOL):
                    if a.conf >= b.conf:
                        keep, drop = a, b
                    else:
                        keep, drop = b, a
                    if drop.conf > keep.best.conf:
                        keep.best = drop.best
                        keep.pan = drop.pan
                        keep.tilt = drop.tilt
                    keep.conf = max(keep.conf, drop.conf)
                    out.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1

    # ---- 类别冲突: pan<6° 且 |Δtilt|<=10° 内异类取高置信 (宁丢 +10 不赌 -20) ----
    out.sort(key=lambda c: -c.conf)
    kept = []
    for c in out:
        conflict = False
        for k in kept:
            if (k.cls != c.cls and
                    abs(k.pan - c.pan) <= CONFLICT_PAN_TOL and
                    abs(k.tilt - c.tilt) <= CONFLICT_TILT_TOL):
                conflict = True
                break
        if not conflict:
            kept.append(c)

    kept.sort(key=lambda c: (c.cls, -c.conf))
    return kept


def choose_aim(dets, img_w, img_h, ppd_h, ppd_v, s_h=1.0, s_v=1.0,
               enemy_conf_min=0.50, danger_conf_min=0.50,
               pan_ok=4.0, tilt_ok=10.0, center_tol=2.0):
    """瞄准到某点后, 基于当前帧检测决定下一步 (开火前安全门)。
    输入 dets: 当前帧 person.onnx 检测列表 [(cls, conf, u, v), ...] (像素)。
    返回 (action, pan_delta, tilt_delta):
      'FIRE'       高置信 enemy 已居中(±center_tol°) → 可以开火
      'ADJUST'     高置信 enemy 在, 但偏 → 需再修正 (pan_delta/tilt_delta 为修正角)
      'NO_ENEMY'   当前视场无高置信 enemy → 不该打
      'SKIP_DANGER'友军/人质落在 ±pan_ok°×±tilt_ok° 内 → 放弃(不赌扣 10)
      'NOTHING'    视场无任何检测
    安全门规则(方案 §3.1.5 收紧):
      - 友军/人质 conf>=danger_conf_min(=0.50) 且在 ±6°pan×±10°tilt 内 → SKIP_DANGER
        (宁可丢掉 +10 敌人分, 也绝不赌 0.5~0.7 置信友军/人质在射程内 -> -10 误伤,
         误伤不封顶, EV 为负)
      - 开火条件: enemy conf>=0.50 且目标居中
    """
    if not dets:
        return ('NOTHING', 0.0, 0.0)
    hx, hy = img_w / 2.0, img_h / 2.0

    for cls, conf, u, v in dets:
        if cls in (HOSTAGE, FRIEND) and conf >= danger_conf_min:
            dp = abs(u - hx) / ppd_h
            dt = abs(v - hy) / ppd_v
            if dp <= pan_ok and dt <= tilt_ok:
                return ('SKIP_DANGER', 0.0, 0.0)

    enemies = [(c, s, u, v) for (c, s, u, v) in dets
               if c == ENEMY and s >= enemy_conf_min]
    if not enemies:
        return ('NO_ENEMY', 0.0, 0.0)

    best = min(enemies, key=lambda d: (d[2] - hx) ** 2 + (d[3] - hy) ** 2)
    dp = s_h * (best[2] - hx) / ppd_h
    dt = s_v * (best[3] - hy) / ppd_v
    if abs(dp) <= center_tol and abs(dt) <= center_tol:
        return ('FIRE', dp, dt)
    return ('ADJUST', dp, dt)
