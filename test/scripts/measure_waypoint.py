#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# measure_waypoint.py —— 航点坐标测量工具
# ------------------------------------------------------------
# 背景: 新图(8/16)与旧航点(8/14 取点)有 ~9cm origin 偏移, 航点坐标偏了。
# 本脚本读当前车在 map 系的位姿 (Cartographer /tracked_pose 优先,
# AMCL /amcl_pose 回退), 采样 5 秒取平均, 输出航点坐标 + yaml 片段,
# 直接替换 nav_competition.yaml 里对应航点的 x/y/yaw_deg。
#
# 用法 (prep 起好后, 车开到目标航点位置、朝向对准目标朝向):
#   rosrun raicom2026 measure_waypoint.py A_pos      # 测 A_pos 航点
#   rosrun raicom2026 measure_waypoint.py            # 不传名, 只打印当前位姿
# ============================================================
import math
import sys
import time

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MeasureWaypoint:
    def __init__(self):
        self._tracked = None   # (x, y, yaw) from /tracked_pose (Cartographer)
        self._amcl = None      # (x, y, yaw) from /amcl_pose (AMCL)
        self._src = None
        rospy.Subscriber('/tracked_pose', PoseStamped, self._cb_tracked)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self._cb_amcl)

    def _cb_tracked(self, msg):
        yaw = quat_to_yaw(msg.pose.orientation)
        self._tracked = (msg.pose.position.x, msg.pose.position.y, yaw)

    def _cb_amcl(self, msg):
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self._amcl = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def run(self, name):
        print('采样 5 秒取平均位姿 ... (车保持静止, 朝向对准目标朝向)')
        samples = []
        src = 'tracked_pose (Cartographer)'
        t0 = time.time()
        while time.time() - t0 < 5.0 and not rospy.is_shutdown():
            p = self._tracked
            if p is None:
                p = self._amcl
                src = 'amcl_pose (AMCL)'
            if p is not None:
                samples.append(p)
            time.sleep(0.1)
        if not samples:
            print('未收到定位位姿! 检查 Cartographer(/tracked_pose) 或 AMCL(/amcl_pose) 是否在跑')
            return
        x = sum(s[0] for s in samples) / len(samples)
        y = sum(s[1] for s in samples) / len(samples)
        yaw = sum(s[2] for s in samples) / len(samples)
        yaw_deg = math.degrees(yaw)
        # 归一化到 (-180, 180]
        while yaw_deg <= -180.0:
            yaw_deg += 360.0
        while yaw_deg > 180.0:
            yaw_deg -= 360.0

        print('')
        print('=' * 52)
        print('  位姿来源: %s' % src)
        print('  车当前位置: x=%.3f  y=%.3f  yaw=%.1f°' % (x, y, yaw_deg))
        print('=' * 52)
        if name:
            print('')
            print('  # 把下面这段替换进 nav_competition.yaml 的 %s 航点:' % name)
            print('  - name: %s' % name)
            print('    x: %.3f' % x)
            print('    y: %.3f' % y)
            print('    yaw_deg: %.1f' % yaw_deg)
            print('')
            print('  ⚠️ yaw_deg 是车"当前朝向", 若车没对准目标朝向请手动改这个值')
        else:
            print('')
            print('  直接坐标: x=%.3f, y=%.3f, yaw_deg=%.1f' % (x, y, yaw_deg))


def main():
    rospy.init_node('measure_waypoint', anonymous=True)
    name = sys.argv[1] if len(sys.argv) > 1 else None
    m = MeasureWaypoint()
    rospy.sleep(0.5)   # 等第一个位姿消息
    m.run(name)


if __name__ == '__main__':
    main()
