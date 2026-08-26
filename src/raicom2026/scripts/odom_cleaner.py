#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# odom_cleaner.py —— 时间戳去重/保序 odometry 转发
# ------------------------------------------------------------
# 为什么需要: cartographer 的 OptimizationProblem2D 要求 odometry
# 时间戳严格递增 (sensor/map_by_time.h MapByTime::Append:
# CHECK_GT(data.time, prev(...)))。一旦上游 (/odom 发布者, 如 EKF)
# 在启动瞬间发出重复或倒退时间戳的消息, cartographer_node 直接 FATAL
# 崩溃 (日志: "Check failed: data.time > std::prev(...)" )。
# EKF/底盘是只读包, 不能改, 所以在本包内做一道过滤:
#   /odom (原样)  ──► 本节点: 只转发时间戳严格递增的消息 ──► /odom_clean
# 丢失的极少数重复帧对里程计/位姿外推无实质影响。
# ------------------------------------------------------------
# 用法 (被 build_map.launch 自动带起; 也可单独跑):
#   rosrun raicom2026 odom_cleaner.py                 # 默认 /odom -> /odom_clean
#   rosrun raicom2026 odom_cleaner.py _in_topic:=/odom _out_topic:=/odom_clean
# ============================================================
import rospy
from nav_msgs.msg import Odometry


class OdomCleaner:
    def __init__(self):
        rospy.init_node('odom_cleaner', anonymous=True)
        in_topic = rospy.get_param('~in_topic', '/odom')
        out_topic = rospy.get_param('~out_topic', '/odom_clean')
        self._last = None                       # 上一条已转发消息的时间戳
        self._dropped = 0
        self._forwarded = 0
        self._pub = rospy.Publisher(out_topic, Odometry, queue_size=10)
        rospy.Subscriber(in_topic, Odometry, self._cb, queue_size=100)
        rospy.loginfo('[odom_cleaner] %s -> %s (仅转发时间戳严格递增的消息)',
                      in_topic, out_topic)

    def _cb(self, msg):
        t = msg.header.stamp
        if t.is_zero():
            return                                  # 丢弃无效(0)时间戳
        if self._last is not None and t <= self._last:
            # 重复或倒退: cartographer 遇此必崩, 直接丢弃
            self._dropped += 1
            if self._dropped <= 5 or self._dropped % 100 == 0:
                rospy.logwarn_throttle(5.0,
                    '[odom_cleaner] 丢弃重复/乱序时间戳 %s (累计 %d 条)',
                    t.to_sec(), self._dropped)
            return
        self._last = t
        self._forwarded += 1
        self._pub.publish(msg)


if __name__ == '__main__':
    try:
        OdomCleaner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
