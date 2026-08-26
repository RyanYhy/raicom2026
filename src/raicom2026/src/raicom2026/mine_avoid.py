# -*- coding: utf-8 -*-
# ============================================================
# raicom2026/mine_avoid.py —— 视觉雷区/反坦克避障 纯逻辑模块
# ------------------------------------------------------------
# visual_mine_avoidance.py 节点 + tests 单测共用, 只依赖 numpy。
# 禁 import: rospy / tf2_ros / tf / image_geometry / torch —— 保证离线可测。
#
# 坐标系约定 (camera_color_optical_frame, 光轴系):
#   x 右, y 下, z 前 (相机光轴)。pinhole 投影直出, 不需要旋转。
# 点云发在传感器帧, 由 costmap_2d 的 C++ tf2 变换到 odom/map,
# 节点内不做任何动态 TF (py3 下 tf2_ros 不可用)。
# ============================================================
import time

import numpy as np


# ============================ 类别常量 ============================
MINE_CLASSES = {0: 'mine_red', 1: 'mine_blue'}   # mine.onnx nc=2
AT_CLASSES = {0: 'anti_tank'}                     # antitank.onnx nc=1

# 各目标球面半径 (m)。costmap 点云用球面点集展开, 任一姿态下 XY 投影≈半径 R 圆盘。
CLASS_RADIUS = {'mine_red': 0.12, 'mine_blue': 0.12, 'anti_tank': 0.15}
# RViz marker 颜色 (r,g,b), 0~1。
CLASS_COLOR = {'mine_red': (1.0, 0.1, 0.1),
               'mine_blue': (0.1, 0.4, 1.0),
               'anti_tank': (1.0, 0.9, 0.1)}


# ============================ 深度 ============================
def depth_mm_to_m(img_mm):
    """16UC1 毫米深度图 → float32 米。NaN/Inf 保持(后续采样处拒绝)。"""
    img = img_mm.astype(np.float32) / 1000.0
    return img


def median_patch(depth_m, dx, dy, pad=1):
    """取 (dx,dy) 周围 (2*pad+1)^2 中值; 越界 clip。返回 float 或 0.0(空 patch)。"""
    h, w = depth_m.shape[:2]
    x0, x1 = max(0, dx - pad), min(w, dx + pad + 1)
    y0, y1 = max(0, dy - pad), min(h, dy + pad + 1)
    patch = depth_m[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.median(patch))


def project_bbox_to_optical(box, fx, fy, cx0, cy0, depth_m,
                            depth_scale_x=1.0, depth_scale_y=400.0 / 480.0,
                            depth_min=0.05, max_range=1.0, depth_patch=3):
    """bbox [left, top, w, h] (color 640x480 原图坐标) → (x,y,z) 米 或 None。

    光轴系直出 (含 '_optical_frame' 不用旋转):
      X = (cx-cx0)/fx*z, Y = (cy-cy0)/fy*z, Z = z
    depth_m 是米制 float32 深度图 (640x400), 采样点按 color/depth 分辨率缩放:
      dx_dep = cx * depth_scale_x, dy_dep = cy * depth_scale_y。
    用 bbox 底边中点采样 (贴地目标底边即与地面接触行)。
    深度无效 (<=depth_min / NaN / Inf) 或超 max_range / Z<0.01 → None。
    """
    left, top, w, h = box
    cx = int(np.clip(left + w / 2.0, 0, 639))
    cy = int(np.clip(top + h, 0, 479))          # 底边
    dx_dep = int(cx * depth_scale_x)             # 640 -> 1.0
    dy_dep = int(cy * depth_scale_y)             # 480 -> 400 (0.8333)
    z = median_patch(depth_m, dx_dep, dy_dep, pad=max(1, depth_patch // 2))
    if not np.isfinite(z) or z <= depth_min:
        return None
    X = (cx - cx0) / fx * z
    Y = (cy - cy0) / fy * z
    if np.hypot(np.hypot(X, Y), z) > max_range or z < 0.01:
        return None
    return (float(X), float(Y), float(z))


# ============================ 点集 ============================
def project_bbox_mono(box, fx, fy, cx0, cy0, disc_m, max_range=1.0):
    """检测框 → 光轴系 3D 点 (纯单目测距, 不用深度图, 2026-08-15)。
    距离 = disc_m * fx / 框宽 (单目相似三角形); pinhole 投影:
      X = (cx-cx0)/fx*d, Y = (cy-cy0)/fy*d, Z = d
    用于反坦克盘避障 —— Orbbec 深度图 depth_registration=False(与彩色
    不对齐)+孔洞+反光, 深度投影会固定在错误位置(车动点云跟车)。
    盘是直立圆盘, 框中心即目标位置。返回 (X,Y,Z) 米或 None。"""
    left, top, w, h = box
    cx = left + w / 2.0
    cy = top + h / 2.0
    bw = max(1.0, w)
    d = disc_m * fx / bw
    if not (0.05 < d < max_range):
        return None
    X = (cx - cx0) / fx * d
    Y = (cy - cy0) / fy * d
    return (float(X), float(Y), float(d))


def make_ring_points(center, radius, n=10):
    """地面圆环点集 (2026-08-14, 参考 github 仿真验证版):
    z 固定 0.01, XY 均匀圆环 n 点 —— 用于 **map 系** (z=上) 点云。
    costmap 的 min/max_obstacle_height 过滤后全部点都保留
    (球面点集 z>0 处只剩顶部小圆, 障碍标记退化成一个点)。
    center=(cx,cy,cz) 米, 返回 [(x,y,z), ...] n 点。"""
    cx, cy, _cz = center
    pts = []
    for i in range(n):
        angle = 2.0 * np.pi * i / float(n)
        pts.append((cx + radius * np.cos(angle),
                    cy + radius * np.sin(angle),
                    0.01))
    return pts


def make_ring_points_optical(center, radius, n=20):
    """地面圆环点集 —— **光轴系** (camera_*_optical_frame: x右 y下 z前)。
    水平地面 = XZ 平面, 所以圆环在 XZ 平面展开, y 保持盘所在高度。
    (2026-08-14 修正: 之前用 XY 平面 -> 竖直立着的圆环, 错)。"""
    cx, cy, cz = center
    pts = []
    for i in range(n):
        angle = 2.0 * np.pi * i / float(n)
        pts.append((cx + radius * np.cos(angle),
                    cy,
                    cz + radius * np.sin(angle)))
    return pts


def make_sphere_points(center, radius, n=20):
    """Fibonacci 球面均匀点集 (确定性, 可测)。保留兼容, 避障默认用圆环。"""
    cx, cy, cz = center
    pts = []
    ga = np.pi * (3.0 - np.sqrt(5.0))            # 黄金角
    for i in range(n):
        yy = 1.0 - (i / float(n - 1)) * 2.0      # [-1, 1]
        rr = np.sqrt(1.0 - yy * yy)
        th = ga * i
        pts.append((cx + radius * np.cos(th) * rr,
                    cy + radius * yy,
                    cz + radius * np.sin(th) * rr))
    return pts


# ============================ 四元数 (单测/扩展用) ============================
def quat_mult(qa, qb):
    """Hamilton 四元数乘法 qa*qb, 四元数 (w, x, y, z)。"""
    wa, xa, ya, za = qa
    wb, xb, yb, zb = qb
    return (wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb)


# ============================ costmap 注入自检 ============================
def check_observation_source(val):
    """参数值 (str) 是否含 'visual_mine' 观察源。非 str / 无 → False。"""
    if not isinstance(val, str):
        return False
    return 'visual_mine' in val.split()


# ============================ 时序确认跟踪器 ============================
class MineTracker:
    """源包 MineTracker 移植 + 增强: 注入 clock (单测用假钟)。

    每类一个实例。detections = [(x,y,z,conf), ...] (光学帧米制坐标)。
    XY 平面匹配 (match_radius), 指数平滑位置, min_hits 确认, max_miss 丢弃,
    stale 秒清理候选 (未确认)。track: {id,pos,hits,miss,state,stamp,conf}。
    """

    def __init__(self, min_hits=2, max_miss=8, match_radius=0.36,
                 smooth=0.5, stale=2.0, clock=time.time):
        self.min_hits = min_hits
        self.max_miss = max_miss
        self.match_radius = match_radius
        self.smooth = smooth
        self.stale = stale
        self.clock = clock
        self.tracks = []
        self._next_id = 0

    def _match(self, x, y):
        """返回命中 track (距离最近且 < match_radius) 或 None。"""
        best, best_d = None, self.match_radius
        for t in self.tracks:
            if t['state'] == 'lost':
                continue
            d = np.hypot(t['pos'][0] - x, t['pos'][1] - y)
            if d < best_d:
                best, best_d = t, d
        return best

    def update(self, detections):
        now = self.clock()
        hit = set()
        for x, y, z, conf in detections:
            t = self._match(x, y)
            if t is None:
                self.tracks.append({'id': self._next_id, 'pos': [x, y, z],
                                    'hits': 1, 'miss': 0, 'state': 'candidate',
                                    'stamp': now, 'conf': conf})
                self._next_id += 1
            else:
                # 指数平滑
                s = self.smooth
                t['pos'] = [s * nv + (1 - s) * ov
                            for ov, nv in zip(t['pos'], (x, y, z))]
                t['hits'] += 1
                t['miss'] = 0
                t['stamp'] = now
                if conf > t['conf']:
                    t['conf'] = conf
                if t['state'] == 'candidate' and t['hits'] >= self.min_hits:
                    t['state'] = 'confirmed'
                hit.add(id(t))
        for t in self.tracks:
            if id(t) not in hit and t['state'] != 'lost':
                t['miss'] += 1
                if t['miss'] >= self.max_miss:
                    t['state'] = 'lost'
        # 清理: lost 立即清; candidate 超 stale 清; confirmed 不因 stale 清 (由 max_miss 控制)
        self.tracks = [t for t in self.tracks
                       if not (t['state'] == 'lost' or
                               (t['state'] == 'candidate' and
                                now - t['stamp'] > self.stale))]
        return len(self.tracks)

    def confirmed_positions(self):
        """返回 [(x,y,z,conf), ...] 已确认 track 的位置。"""
        return [(t['pos'][0], t['pos'][1], t['pos'][2], t['conf'])
                for t in self.tracks if t['state'] == 'confirmed']
