#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# test_visual_mine_avoidance.py —— 视觉雷区避障 纯逻辑离线单测
# ------------------------------------------------------------
# 不依赖 rospy / 相机 / 模型, 直接用 yolov8 环境跑:
#   /home/mowen/miniconda3/envs/yolov8/bin/python tests/test_visual_mine_avoidance.py
# 通过则打印 "ALL TESTS PASSED" 并以 0 退出。
#
# 覆盖: 深度mm→m / bbox→光学坐标 / 无效深度拒绝 / 球面点集免疫旋转 /
#       MineTracker 确认·丢弃·平滑·匹配·每类独立·stale 清理 /
#       CLASS_* 完整性 / costmap 注入自检。
# ============================================================
import ast
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from raicom2026.mine_avoid import (
    MINE_CLASSES, AT_CLASSES, CLASS_RADIUS, CLASS_COLOR,
    depth_mm_to_m, median_patch, project_bbox_to_optical,
    make_sphere_points, quat_mult, check_observation_source,
    MineTracker,
)

FAILED = []


def check(name, cond):
    if cond:
        print('  [OK] %s' % name)
    else:
        print('  [FAIL] %s' % name)
        FAILED.append(name)


class FakeClock:
    """注入 MineTracker 的假时钟, 完全确定性。"""
    def __init__(self, t0=0.0):
        self.t = t0

    def __call__(self):
        return self.t


def test_depth():
    print('== 深度 毫米→米 ==')
    d = depth_mm_to_m(np.array([[1000, 2000], [0, 1500]], np.uint16))
    check('毫米→米 数值', np.allclose(d, [[1.0, 2.0], [0.0, 1.5]]))
    check('返回 float32', d.dtype == np.float32)
    # median_patch 越界 clip: 完全越界 -> 空 patch -> 0.0 (不崩)
    img = np.full((400, 640), 0.5, np.float32)
    check('median_patch 完全越界返回0.0', median_patch(img, 700, 500) == 0.0)
    # 边缘 clip: pad 收缩到图内, patch 非空取中值 (不崩)
    # 注意 float32 精度: float(np.float32(0.9)) != 0.9, 用容差比较
    img[398:400, 638:640] = 0.9
    check('median_patch 边缘clip取中值', abs(median_patch(img, 639, 399) - 0.9) < 1e-6)


def test_project_bbox():
    print('== bbox→光学坐标 ==')
    fx, fy, cx0, cy0 = 610.0, 610.0, 320.0, 240.0
    # 深 0.6m, 像素 (320,300) -> (0.0, 0.0590, 0.6)
    depth_m = np.full((400, 640), 0.6, np.float32)
    box = [300, 280, 40, 20]                 # cx=320, cy=300 (底边)
    pt = project_bbox_to_optical(box, fx, fy, cx0, cy0, depth_m)
    check('数值(0.0,0.0590,0.6)',
          pt is not None and abs(pt[0] - 0.0) < 1e-6 and
          abs(pt[1] - (300 - cy0) / fy * 0.6) < 1e-6 and abs(pt[2] - 0.6) < 1e-6)

    # depth y 400/480 缩放 + 3x3 中值拒绝离群
    d2 = np.full((400, 640), 0.5, np.float32)
    # 采样点 (dx=320, dy=250) 的 3x3: 0.7 环绕, 中心 5.0 离群
    d2[249:252, 319:322] = 0.7
    d2[250, 320] = 5.0
    pt_scaled = project_bbox_to_optical(box, fx, fy, cx0, cy0, d2)
    pt_unscaled = project_bbox_to_optical(box, fx, fy, cx0, cy0, d2, depth_scale_y=1.0)
    check('y 缩放采样到 0.7 (中值滤离群)', pt_scaled is not None and abs(pt_scaled[2] - 0.7) < 1e-6)
    check('不缩放则采到 0.5 (证明缩放有效)',
          pt_unscaled is not None and abs(pt_unscaled[2] - 0.5) < 1e-6)

    # 无效深度拒绝
    d_near = np.full((400, 640), 0.01, np.float32)
    check('z<=depth_min 拒绝', project_bbox_to_optical(box, fx, fy, cx0, cy0, d_near) is None)
    d_nan = np.full((400, 640), np.nan, np.float32)
    check('NaN 拒绝', project_bbox_to_optical(box, fx, fy, cx0, cy0, d_nan) is None)
    d_inf = np.full((400, 640), np.inf, np.float32)
    check('Inf 拒绝', project_bbox_to_optical(box, fx, fy, cx0, cy0, d_inf) is None)
    # 超 max_range
    d_far = np.full((400, 640), 0.6, np.float32)
    check('超 max_range 拒绝',
          project_bbox_to_optical(box, fx, fy, cx0, cy0, d_far, max_range=0.3) is None)
    # Z < 0.01 (depth_min 放开后仍拒)
    d_tiny = np.full((400, 640), 0.005, np.float32)
    check('Z<0.01 拒绝',
          project_bbox_to_optical(box, fx, fy, cx0, cy0, d_tiny, depth_min=0.0) is None)


def _rotate(points, q):
    """四元数旋转点集 (q 为单位四元数 w,x,y,z)。"""
    qc = (q[0], -q[1], -q[2], -q[3])
    out = []
    for (x, y, z) in points:
        p = quat_mult(quat_mult(q, (0.0, x, y, z)), qc)
        out.append(p[1:])
    return out


def _octant_coverage(pts, R, frac=0.6):
    """8 个 45° 方位扇区各取最大 XY 半径, 是否全部 >= frac*R。"""
    for k in range(8):
        lo, hi = math.radians(k * 45), math.radians((k + 1) * 45)
        best = 0.0
        for (x, y, _z) in pts:
            r = math.hypot(x, y)
            a = math.atan2(y, x)
            if a < 0:
                a += 2 * math.pi
            if lo <= a < hi:
                best = max(best, r)
        if best < frac * R:
            return False
    return True


def test_sphere():
    print('== 球面点集 免疫姿态 ==')
    R = 0.12
    pts = make_sphere_points((0.5, -0.3, 0.7), R, 20)
    check('点数正确', len(pts) == 20)
    check('半径正确', all(abs(math.hypot(math.hypot(x - 0.5, y + 0.3), z - 0.7) - R) < 1e-9
                          for x, y, z in pts))
    # 三个旋转 (绕z 90°, 绕x 90°, 任意单位四元数) 后 XY 投影仍覆盖圆盘
    qzs = math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)
    qxs = math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0
    n = math.sqrt(0.6 ** 2 + 0.5 ** 2 + 0.3 ** 2 + 0.2 ** 2)
    qa = tuple(v / n for v in (0.6, 0.5, 0.3, 0.2))
    for name, q in [('绕z90°', qzs), ('绕x90°', qxs), ('任意', qa)]:
        rp = _rotate([(p[0] - 0.5, p[1] + 0.3, p[2] - 0.7) for p in pts], q)
        maxr = max(math.hypot(x, y) for x, y, _ in rp)
        check('%s: 赤道 XY 半径≈R' % name, maxr >= 0.95 * R)
        check('%s: 8 扇区覆盖 0.6R' % name, _octant_coverage(rp, R))


def test_tracker():
    print('== MineTracker 时序确认 ==')
    clk = FakeClock()
    tr = MineTracker(min_hits=2, max_miss=8, match_radius=0.36, smooth=0.5,
                     stale=2.0, clock=clk)
    # 确认: 两次命中 -> confirmed
    tr.update([(0.0, 0.0, 1.0, 0.8)])
    check('一次命中未确认', len(tr.confirmed_positions()) == 0)
    tr.update([(0.0, 0.0, 1.0, 0.8)])
    cf = tr.confirmed_positions()
    check('两次命中已确认', len(cf) == 1 and abs(cf[0][0]) < 1e-9 and abs(cf[0][1]) < 1e-9)
    check('置信取最大', cf and cf[0][3] == 0.8)
    # 丢弃: 连续 8 次空 -> lost 并清除
    for _ in range(8):
        tr.update([])
    check('连续8次miss丢弃', len(tr.confirmed_positions()) == 0 and len(tr.tracks) == 0)

    # 平滑: s=0.5, 0 -> 0.2 移动后半程位置
    clk2 = FakeClock()
    tr2 = MineTracker(min_hits=2, max_miss=8, match_radius=0.36, smooth=0.5,
                      stale=2.0, clock=clk2)
    tr2.update([(0.0, 0.0, 1.0, 0.5)])
    tr2.update([(0.2, 0.0, 1.0, 0.5)])
    cf2 = tr2.confirmed_positions()
    check('指数平滑 pos=(0.1,0,1)', len(cf2) == 1 and
          abs(cf2[0][0] - 0.1) < 1e-9 and abs(cf2[0][2] - 1.0) < 1e-9)

    # 匹配: 两目标相距 > match_radius -> 两条独立 track
    clk3 = FakeClock()
    tr3 = MineTracker(min_hits=1, match_radius=0.36, clock=clk3)
    tr3.update([(0.0, 0.0, 1.0, 0.8), (0.5, 0.0, 1.0, 0.7)])
    tr3.update([(0.0, 0.0, 1.0, 0.8), (0.5, 0.0, 1.0, 0.7)])   # 二次命中 -> 各自 confirmed
    check('两目标两条track', len(tr3.confirmed_positions()) == 2)
    # 同目标跨帧匹配(相距 < match_radius) -> 不新增 track
    tr3.update([(0.03, 0.02, 1.0, 0.8)])
    check('近邻跨帧仍一条track', len(tr3.tracks) == 2)

    # stale 清理 candidate (未确认)
    clk4 = FakeClock()
    tr4 = MineTracker(min_hits=5, match_radius=0.36, stale=2.0, clock=clk4)
    tr4.update([(0.0, 0.0, 1.0, 0.8)])          # candidate, t=0
    clk4.t = 1.0
    tr4.update([])                               # 未确认且未超时, 保留
    check('未超时保留', len(tr4.tracks) == 1)
    clk4.t = 3.0
    tr4.update([])                               # 超 stale -> candidate 清除
    check('超时清除 candidate', len(tr4.tracks) == 0)

    # confirmed 不因 stale 清除 (由 max_miss 控制), 保证点云持续注入
    clk5 = FakeClock()
    tr5 = MineTracker(min_hits=2, max_miss=8, match_radius=0.36, stale=2.0, clock=clk5)
    tr5.update([(0.0, 0.0, 1.0, 0.8)])
    tr5.update([(0.0, 0.0, 1.0, 0.8)])          # confirmed
    clk5.t = 10.0
    tr5.update([])
    check('confirmed 不因 stale 清除', len(tr5.confirmed_positions()) == 1)

    # 每类独立
    a = MineTracker(min_hits=1, clock=FakeClock())
    b = MineTracker(min_hits=1, clock=FakeClock())
    a.update([(0.0, 0.0, 1.0, 0.8)])
    a.update([(0.0, 0.0, 1.0, 0.8)])
    check('每类独立互不影响', len(a.confirmed_positions()) == 1 and len(b.tracks) == 0)


def test_class_table():
    print('== 类别表完整性 ==')
    for key in list(MINE_CLASSES.values()) + list(AT_CLASSES.values()):
        check('半径/颜色齐全(%s)' % key,
              key in CLASS_RADIUS and key in CLASS_COLOR and
              0 < CLASS_RADIUS[key] and all(0 <= c <= 1 for c in CLASS_COLOR[key]))
    check('mine 2类', len(MINE_CLASSES) == 2 and 0 in MINE_CLASSES and 1 in MINE_CLASSES)
    check('antitank 1类', len(AT_CLASSES) == 1)


def test_obs_source():
    print('== costmap 注入自检 ==')
    check('含 visual_mine -> True', check_observation_source('scan visual_mine') is True)
    check('仅 scan -> False', check_observation_source('scan') is False)
    check('子串不算(visual_mine_fake) -> False',
          check_observation_source('scan visual_mine_fake') is False)
    check('非 str -> False', check_observation_source(123) is False and
          check_observation_source(None) is False and
          check_observation_source(['visual_mine']) is False)


def test_no_forbidden_imports():
    print('== mine_avoid 无 py3 禁用依赖 ==')
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'src', 'raicom2026', 'mine_avoid.py'),
               encoding='utf-8').read()
    # 用 AST 精确检查真实 import, 避免误判注释/文档字符串里的字面词
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    forbidden = ('rospy', 'tf2_ros', 'tf', 'image_geometry', 'torch')
    bad = [i for i in imports
           if any(i == f or i.startswith(f + '.') for f in forbidden)]
    check('无禁用 import (%s)' % (','.join(bad) or '-'), not bad)


if __name__ == '__main__':
    test_depth()
    test_project_bbox()
    test_sphere()
    test_tracker()
    test_class_table()
    test_obs_source()
    test_no_forbidden_imports()
    print()
    if FAILED:
        print('FAILED %d: %s' % (len(FAILED), ', '.join(FAILED)))
        sys.exit(1)
    print('ALL TESTS PASSED')
