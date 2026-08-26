#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# mark_waypoint_batch.py —— 批量标航点 (地图点击, 一次点完所有点)
# ------------------------------------------------------------
# 运行一次, 按顺序连续点击地图上的各个航点, 每个点击立即输出坐标,
# 全部点完后一次性输出完整 yaml 片段 (可直接替换 course)。
# yaw_deg 自动从 param/nav_competition.yaml 读旧值 (origin 偏移是平移,
# 不影响朝向, 旧 yaw 直接沿用)。
#
# 用法:
#   1) prep 起好 + rviz 打开 (Fixed Frame = map)
#   2) 标所有航点:  rosrun raicom2026 mark_waypoint_batch.py
#      只标任务点:  rosrun raicom2026 mark_waypoint_batch.py A_pos B_pos MineA_pos C_pos D_pos MineB_pos IV_pos
#   3) rviz 点 "Publish Point", 按提示逐个点击地图上的航点位置
#   4) 点完最后一个, 脚本输出完整 yaml 片段
# ============================================================
import os
import sys

import rospy
import yaml
from geometry_msgs.msg import PointStamped


def load_course():
    """从 param/nav_competition.yaml 读 course, 返回 (顺序航点名列表, name映射)。"""
    try:
        import rospkg
        pkg = rospkg.RosPack().get_path('raicom2026')
    except Exception:
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(pkg, 'param', 'nav_competition.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    course = cfg.get('course', [])
    ordered = [wp.get('name') for wp in course if wp.get('name')]
    m = {}
    for wp in course:
        m[wp.get('name')] = {
            'x': wp.get('x'), 'y': wp.get('y'),
            'yaw_deg': wp.get('yaw_deg'),
        }
    return ordered, m


def main():
    names = sys.argv[1:]
    ordered, course = load_course()
    if not names:
        # 默认: 标所有航点 (按 course 顺序, 含过程点)
        names = ordered
        print('未指定航点, 默认标全部 %d 个航点 (含过程点):' % len(names))
        print('  %s' % ', '.join(names))
        print('(只标部分点, 运行时把航点名作为参数传入)')

    rospy.init_node('mark_waypoint_batch', anonymous=True)

    results = []
    print('')
    print('=' * 56)
    print('  批量标航点: 共 %d 个点' % len(names))
    print('  每次提示后, 用 rviz "Publish Point" 点击对应位置')
    print('  (Fixed Frame 必须是 map)')
    print('=' * 56)

    for i, name in enumerate(names, 1):
        old = course.get(name, {})
        old_yaw = old.get('yaw_deg')
        print('')
        print('[%d/%d] 请点击 %s (旧坐标 x=%.2f y=%.2f, yaw 沿用 %.1f°)' % (
            i, len(names), name,
            old.get('x', 0), old.get('y', 0),
            old_yaw if old_yaw is not None else 0))

        try:
            msg = rospy.wait_for_message('/clicked_point', PointStamped, timeout=180.0)
        except rospy.ROSException:
            print('  超时未点击, 跳过 %s (可重新运行)' % name)
            results.append(None)
            continue

        x = round(msg.point.x, 3)
        y = round(msg.point.y, 3)
        results.append({'name': name, 'x': x, 'y': y, 'yaw_deg': old_yaw})
        print('    -> x=%.3f  y=%.3f  yaw=%.1f°' % (x, y, old_yaw if old_yaw is not None else 0))

    # 输出完整 yaml 片段
    print('')
    print('=' * 56)
    print('  完整 yaml 片段 (替换 nav_competition.yaml 对应航点):')
    print('=' * 56)
    for r in results:
        if r is None:
            continue
        yaw = r['yaw_deg']
        print('  - name: %s' % r['name'])
        print('    x: %.3f' % r['x'])
        print('    y: %.3f' % r['y'])
        if yaw is not None:
            print('    yaw_deg: %s' % yaw)
    print('=' * 56)
    print('')


if __name__ == '__main__':
    main()
