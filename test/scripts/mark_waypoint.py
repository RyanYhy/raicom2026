#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# mark_waypoint.py —— 在地图上直接标航点 (不用开车)
# ------------------------------------------------------------
# 原理: 航点坐标 = map 系 x/y, 地图(pgm) 就是 map 系的图。rviz 里
#   "Publish Point" 点击地图, 输出就是 map 系坐标, 和开车实测完全等价,
#   且不受定位误差影响。
#
# 用法:
#   1) prep 起好, rviz 打开 (Fixed Frame = map, 能看到地图墙线)
#   2) 终端运行: rosrun raicom2026 mark_waypoint.py A_pos 147.9
#      (参数1=航点名, 参数2=该航点 yaw_deg, 用旧航点的值即可)
#   3) rviz 工具栏点 "Publish Point" (十字准星), 再点击地图上的航点位置
#   4) 脚本输出 x/y + 完整 yaml 片段
#
# 批量标点: 逐个航点重复 2-4 步 (每次运行 + 点击)。
# 注意: origin 偏移是平移, yaw_deg 不受影响, 旧航点的 yaw 直接沿用。
# ============================================================
import sys

import rospy
from geometry_msgs.msg import PointStamped


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    yaw = sys.argv[2] if len(sys.argv) > 2 else None

    rospy.init_node('mark_waypoint', anonymous=True)

    label = ('航点 %s' % name) if name else '航点'
    print('')
    print('=' * 52)
    print('  请用 rviz 的 "Publish Point" 工具点击地图上的 %s 位置' % label)
    print('  (Fixed Frame 必须是 map, 否则坐标是错的)')
    print('=' * 52)

    try:
        msg = rospy.wait_for_message('/clicked_point', PointStamped, timeout=120.0)
    except rospy.ROSException:
        print('120 秒未收到点击! 确认 rviz 用了 Publish Point 工具')
        return

    x = msg.point.x
    y = msg.point.y
    print('')
    print('=' * 52)
    print('  点击坐标: x=%.3f  y=%.3f' % (x, y))
    print('=' * 52)
    if name and yaw:
        print('')
        print('  # 替换进 nav_competition.yaml 的 %s 航点:' % name)
        print('  - name: %s' % name)
        print('    x: %.3f' % x)
        print('    y: %.3f' % y)
        print('    yaw_deg: %s' % yaw)
        print('')
    elif name:
        print('')
        print('  - name: %s' % name)
        print('    x: %.3f' % x)
        print('    y: %.3f' % y)
        print('    yaw_deg: <沿用旧值>' % y)
        print('')
    else:
        print('')
        print('  坐标: x=%.3f, y=%.3f' % (x, y))


if __name__ == '__main__':
    main()
