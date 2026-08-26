#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# test_nav_competition.py —— 自主导航编排 纯逻辑离线单测
# ------------------------------------------------------------
# 不依赖 rospy / move_base / 相机, 直接用 yolov8 环境跑:
#   /home/mowen/miniconda3/envs/yolov8/bin/python tests/test_nav_competition.py
# 通过则打印 "ALL TESTS PASSED" 并以 0 退出。
# ============================================================
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from raicom2026.nav_logic import (
    yaw_deg_to_quat, validate_course, lateral_offset_goals, Budget,
    pick_status_zone,
)

FAILED = []


def check(name, cond):
    if cond:
        print('  [OK] %s' % name)
    else:
        print('  [FAIL] %s' % name)
        FAILED.append(name)


# ---------------- yaw -> quat ----------------
def test_yaw_to_quat():
    print('== yaw_deg_to_quat ==')
    x, y, z, w = yaw_deg_to_quat(0.0)
    check('0° -> (0,0,0,1)', (x, y) == (0.0, 0.0) and z == 0.0 and w == 1.0)
    x, y, z, w = yaw_deg_to_quat(90.0)
    check('90° z=w=sin45', abs(z - math.sqrt(2) / 2) < 1e-9 and abs(w - math.sqrt(2) / 2) < 1e-9)
    x, y, z, w = yaw_deg_to_quat(180.0)
    check('180° -> (0,0,1,0)', abs(z - 1.0) < 1e-9 and abs(w) < 1e-9)
    x, y, z, w = yaw_deg_to_quat(45.0)
    check('四元数模=1', abs(x * x + y * y + z * z + w * w - 1.0) < 1e-9)


# ---------------- validate_course ----------------
def _good_wp(name, x, y, yaw, tasks=None, **kw):
    d = {'name': name, 'x': x, 'y': y, 'yaw_deg': yaw, 'tasks': tasks or []}
    d.update(kw)
    return d


def test_validate_course():
    print('== validate_course ==')
    good = [
        _good_wp('A', 1.0, 0.0, 90, [{'type': 'person', 'zone': 'A'}]),
        _good_wp('B', 2.0, 1.0, 90, [{'type': 'person', 'zone': 'B'},
                                     {'type': 'mine', 'zone': 'A'}]),
        _good_wp('C', 3.0, 2.0, 90, [{'type': 'person', 'zone': 'C'},
                                     {'type': 'mine', 'zone': 'B'}]),
        _good_wp('D', 4.0, 3.0, 90, [{'type': 'person', 'zone': 'D'}], iv=True),
        _good_wp('END', 0.0, 0.0, 0),
    ]
    check('合法课程通过', validate_course(good) is True)
    check('空课程拒绝', _raises(validate_course, []))
    check('非列表拒绝', _raises(validate_course, 'x'))
    check('缺name拒绝', _raises(validate_course, [{'x': 1, 'y': 2, 'yaw_deg': 0, 'tasks': []}]))
    check('缺x拒绝', _raises(validate_course, [_good_wp('A', None, 2, 0)]))
    dup = [dict(good[0]), dict(good[0])]
    check('重复name拒绝', _raises(validate_course, dup))
    badtype = [dict(good[0])]
    badtype[0]['tasks'] = [{'type': 'shoot', 'zone': 'A'}]
    check('非法任务类型拒绝', _raises(validate_course, badtype))
    badzone = [dict(good[0])]
    badzone[0]['tasks'] = [{'type': 'person', 'zone': 'E'}]
    check('非法人物区拒绝', _raises(validate_course, badzone))
    badmine = [dict(good[1])]
    badmine[0]['tasks'] = [{'type': 'mine', 'zone': 'C'}]
    check('非法雷区拒绝', _raises(validate_course, badmine))
    # 人物A 重复触发
    dupp = [dict(good[0]), _good_wp('X', 9, 9, 0, [{'type': 'person', 'zone': 'A'}])]
    check('人物区重复触发拒绝', _raises(validate_course, dupp))
    # 人物A~D 未全触发仍合法 (允许漏做某区, 报告补 0)
    partial = [dict(good[0]), dict(good[4])]
    check('部分区合法(报告补0)', validate_course(partial) is True)


def _raises(fn, *a):
    try:
        fn(*a)
        return False
    except ValueError:
        return True


# ---------------- lateral_offset_goals ----------------
def test_lateral_offsets():
    print('== lateral_offset_goals ==')
    from_wp = {'x': 0.0, 'y': 0.0}
    to_wp = {'x': 1.0, 'y': 0.0}
    offs = lateral_offset_goals(from_wp, to_wp, 0.35)
    check('得到2个横向目标', len(offs) == 2)
    (x1, y1), (x2, y2) = offs
    check('横向垂直于走向(朝±y)', abs(y1 - 0.35) < 1e-9 and abs(y2 + 0.35) < 1e-9)
    check('x保持不变(=目标x)', abs(x1 - 1.0) < 1e-9 and abs(x2 - 1.0) < 1e-9)
    check('from==to -> 空', lateral_offset_goals({'x': 1, 'y': 1}, {'x': 1, 'y': 1}, 0.35) == [])
    check('距离过近 -> 空', lateral_offset_goals({'x': 0, 'y': 0}, {'x': 1e-12, 'y': 0}, 0.35) == [])


# ---------------- Budget ----------------
class FakeClock:
    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t


def test_budget():
    print('== Budget ==')
    clk = FakeClock(0.0)
    b = Budget(280, now=clk)
    check('初始未耗尽', not b.exhausted() and b.remaining() == 280)
    clk.t = 100.0
    check('100s 未耗尽', not b.exhausted())
    clk.t = 280.0
    check('280s 恰好耗尽', b.exhausted())
    clk.t = 300.0
    check('300s 超预算', b.exhausted() and b.remaining() <= 0)


# ---------------- pick_status_zone ----------------
def test_pick_status_zone():
    print('== pick_status_zone ==')
    check('人物A', pick_status_zone('A区发现2个敌军、1个友军和0个人质', 'person') == 'A')
    check('雷区B(半角逗号)', pick_status_zone('B雷区未排雷数量0个,已经排雷数量1个', 'mine') == 'B')
    check('人物非法串 -> None', pick_status_zone('E区发现...', 'person') is None)
    check('雷区非法串 -> None', pick_status_zone('C雷区...', 'mine') is None)
    check('空串 -> None', pick_status_zone('', 'person') is None)
    check('None -> None', pick_status_zone(None, 'person') is None)


if __name__ == '__main__':
    test_yaw_to_quat()
    test_validate_course()
    test_lateral_offsets()
    test_budget()
    test_pick_status_zone()
    print()
    if FAILED:
        print('FAILED: %s' % ', '.join(FAILED))
        sys.exit(1)
    print('ALL TESTS PASSED')
