#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# mark_waypoint_live.py —— 实测标航点工具 (车开到位置, 回车记录)
# ------------------------------------------------------------
# 在真实场地重新测全部航点。车开到每个航点位置、朝向对准,
# 按回车记录当前定位位姿, 支持任意数量航点 (可多可少)。
#
# 用法 (prep 起好 + 定位正常 + 车能开到场地各位置):
#   rosrun raicom2026 mark_waypoint_live.py
#
# 交互流程:
#   1) 把车开到航点位置, 车头朝向对准目标朝向, 停稳
#   2) 按回车 -> 采样 3 秒取平均位姿 (车保持静止)
#   3) 输入航点名 (直接回车 = 自动编号 P1/P2/...)
#   4) 重复, 全部记录完输入 q 结束
#   5) 输出完整 yaml 片段 (替换 nav_competition.yaml 的 course)
#
# 位姿来源: Cartographer /tracked_pose 优先, AMCL /amcl_pose 回退。
# ============================================================
import math
import time

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LiveMarker:
    def __init__(self):
        self._tracked = None
        self._amcl = None
        self._src = 'tracked_pose'
        rospy.Subscriber('/tracked_pose', PoseStamped, self._cb_tracked)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self._cb_amcl)

    def _cb_tracked(self, msg):
        yaw = quat_to_yaw(msg.pose.orientation)
        self._tracked = (msg.pose.position.x, msg.pose.position.y, yaw)

    def _cb_amcl(self, msg):
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self._amcl = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def sample(self, secs=3.0):
        """采样 secs 秒取平均位姿。返回 (x, y, yaw_deg) 或 None。"""
        samples = []
        t0 = time.time()
        while time.time() - t0 < secs and not rospy.is_shutdown():
            p = self._tracked
            if p is not None:
                samples.append(p)
                self._src = 'tracked_pose'
            else:
                p = self._amcl
                if p is not None:
                    samples.append(p)
                    self._src = 'amcl_pose'
            time.sleep(0.1)
        if not samples:
            return None
        x = sum(s[0] for s in samples) / len(samples)
        y = sum(s[1] for s in samples) / len(samples)
        yaw = sum(s[2] for s in samples) / len(samples)
        yaw_deg = math.degrees(yaw)
        while yaw_deg <= -180.0:
            yaw_deg += 360.0
        while yaw_deg > 180.0:
            yaw_deg -= 360.0
        return (x, y, yaw_deg)

    def run(self):
        waypoints = []
        idx = 0
        print('')
        print('=' * 56)
        print('  实测标航点 (任意数量)')
        print('  操作: 车开到航点+朝向对准+停稳 -> 回车记录')
        print('  位姿来源: %s' % self._src)
        print('=' * 56)
        while True:
            idx += 1
            try:
                s = input('[%d] 车停好后按回车记录 (q=结束, 其他字符=自定义名): ' % idx)
            except (EOFError, KeyboardInterrupt):
                break
            if s.strip().lower() in ('q', 'quit', 'exit', 'done'):
                break
            try:
                print('  采样中 ... (车保持静止)')
                pose = self.sample(3.0)
            except KeyboardInterrupt:
                break
            if pose is None:
                print('  ⚠️ 未收到定位位姿 (检查 /tracked_pose 或 /amcl_pose)')
                continue
            x, y, yaw = pose
            name = s.strip() if s.strip() else 'P%d' % idx
            waypoints.append({'name': name, 'x': x, 'y': y, 'yaw_deg': yaw})
            print('  ✅ %s: x=%.3f y=%.3f yaw=%.1f° (%s)'
                  % (name, x, y, yaw, self._src))

        if not waypoints:
            print('未记录任何航点')
            return
        print('')
        print('=' * 56)
        print('  共 %d 个航点, yaml 片段 (替换 nav_competition.yaml 的 course):' % len(waypoints))
        print('=' * 56)
        for wp in waypoints:
            print('  - name: %s' % wp['name'])
            print('    x: %.3f' % wp['x'])
            print('    y: %.3f' % wp['y'])
            print('    yaw_deg: %.1f' % wp['yaw_deg'])
        print('=' * 56)
        print('')


def main():
    rospy.init_node('mark_waypoint_live', anonymous=True)
    rospy.sleep(1.0)
    LiveMarker().run()


if __name__ == '__main__':
    main()
