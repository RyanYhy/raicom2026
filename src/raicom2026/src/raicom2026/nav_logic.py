#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# raicom2026/nav_logic.py —— 自主导航编排的纯逻辑
# ------------------------------------------------------------
# nav_competition.py 的状态机/航点/预算的**可离线单测**部分。
# 只依赖标准库, 不依赖 rospy / actionlib, 可脱离硬件跑
#   tests/test_nav_competition.py。
# 涵盖:
#   - 航点课程 (course) 的结构校验 (name/x/y/yaw_deg/tasks)
#   - yaw(度)->四元数 (绕 Z, roll=pitch=0)
#   - 横向绕行目标计算 (Ⅳ 走廊反坦克兜底用)
#   - 全局时间预算 Budget (可注入时钟便于单测)
# ============================================================
import math
import time

PERSON_ZONES = 'ABCD'
MINE_ZONES = 'AB'
VALID_TASK_TYPES = ('person', 'mine', 'avoid_on', 'avoid_off', 'antitank')
_REQUIRED_WP_KEYS = ('x', 'y', 'yaw_deg')


def yaw_deg_to_quat(yaw_deg):
    """绕 Z 轴 yaw(度) -> (x, y, z, w)。地面导航 roll=pitch=0。"""
    y = math.radians(float(yaw_deg))
    return (0.0, 0.0, math.sin(y / 2.0), math.cos(y / 2.0))


def validate_course(course):
    """校验比赛航点课程。非法直接抛 ValueError (宁可启动即报错, 不带病开车)。"""
    if not isinstance(course, list) or not course:
        raise ValueError('course 必须是非空列表')
    seen_names = set()
    seen_person = set()
    seen_mine = set()
    for i, wp in enumerate(course):
        if not isinstance(wp, dict):
            raise ValueError('航点 %d 必须是 dict' % i)
        name = wp.get('name')
        if not isinstance(name, str) or not name:
            raise ValueError('航点 %d 缺 name' % i)
        if name in seen_names:
            raise ValueError('航点 name 重复: %r' % name)
        seen_names.add(name)
        for k in _REQUIRED_WP_KEYS:
            v = wp.get(k)
            if not isinstance(v, (int, float)):
                raise ValueError('航点 %s 缺数值字段 %s' % (name, k))
        tasks = wp.get('tasks', [])
        if not isinstance(tasks, list):
            raise ValueError('航点 %s 的 tasks 必须是列表' % name)
        for t in tasks:
            if not isinstance(t, dict):
                raise ValueError('航点 %s 的 task 必须是 dict' % name)
            tt = t.get('type')
            z = t.get('zone')
            if tt not in VALID_TASK_TYPES:
                raise ValueError('航点 %s 非法任务类型 %r' % (name, tt))
            if tt == 'person':
                if z not in PERSON_ZONES:
                    raise ValueError('航点 %s 非法人物区 %r' % (name, z))
                if z in seen_person:
                    raise ValueError('人物区 %s 重复触发' % z)
                seen_person.add(z)
            elif tt == 'mine':
                if z not in MINE_ZONES:
                    raise ValueError('航点 %s 非法雷区 %r' % (name, z))
                if z in seen_mine:
                    raise ValueError('雷区 %s 重复触发' % z)
                seen_mine.add(z)
    return True


def lateral_offset_goals(from_wp, to_wp, offset_m):
    """返回两个横向绕行目标 (x,y) 列表, 方向垂直于 from->to 段, 偏移 ±offset_m。
    用于 Ⅳ 走廊反坦克盘兜底: move_base 主路径被盘堵死时, 朝走廊横向空间发临时目标,
    由规划器绕行。from==to 或距离过近返回空列表 (无法求垂直方向)。"""
    dx = float(to_wp['x']) - float(from_wp['x'])
    dy = float(to_wp['y']) - float(from_wp['y'])
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return []
    px, py = -dy / length, dx / length
    return [
        (to_wp['x'] + px * offset_m, to_wp['y'] + py * offset_m),
        (to_wp['x'] - px * offset_m, to_wp['y'] - py * offset_m),
    ]


class Budget:
    """全局时间预算。now 可注入 (单测传假时钟), 实机用默认 time.time。"""

    def __init__(self, limit_s, now=None):
        self.limit_s = float(limit_s)
        self._now = now or time.time
        self.t0 = self._now()

    def elapsed(self):
        return self._now() - self.t0

    def remaining(self):
        return self.limit_s - self.elapsed()

    def exhausted(self):
        return self.elapsed() >= self.limit_s


def pick_status_zone(line, kind):
    """从字节精确状态串里取区名; 不匹配返回 None。
    person: "A区发现..." 首字符 A~D
    mine  : "A雷区未排雷..." 首字符 A~B
    (report.py 的正则已保证串格式, 这里只取首字符做触发匹配)"""
    if not isinstance(line, str) or not line:
        return None
    c = line[0]
    if kind == 'person' and c in PERSON_ZONES:
        return c
    if kind == 'mine' and c in MINE_ZONES:
        return c
    return None
