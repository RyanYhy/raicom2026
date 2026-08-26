#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# cmd_vel_merge.py —— 循线横向修正叠加节点 (2026-08-14)
# ------------------------------------------------------------
# 比赛场地车道黄线循迹辅助:
#   move_base 规划输出 (线速度+角速度) 不变, 叠加循线节点的横向修正
#   (麦克纳姆轮 vy) 后发布到 /cmd_vel。
#
# 数据流:
#   move_base --/move_base/cmd_vel--> [本节点] --/cmd_vel--> 底盘
#   visual_mine_avoidance --/visual_line/deviation+valid--> [本节点]
#
# 修正规则:
#   - valid=false (当前视野无线) -> 修正 0, 纯 move_base 输出
#   - 线偏右 (dev>0) -> 车向右移 vy<0 (ROS y 轴朝左)
#   - vy 限幅 ±VY_MAX (m/s), 防止循线盖过避障/导航
#   - 仅在线速度 >0.03 时启用 (车停着不需要修正)
# 参数 (rosparam ~ 前缀):
#   ~kp (默认 1.0):  vy = -kp * deviation_m
#   ~vy_max (默认 0.08): 横向修正速度限幅
# ============================================================
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool


class CmdVelMerge:
    def __init__(self):
        self.kp = rospy.get_param('~kp', 1.0)
        self.vy_max = rospy.get_param('~vy_max', 0.08)
        self.min_vx = rospy.get_param('~min_vx', 0.03)
        # 2026-08-15: 循线修正开关, 默认关 (符号未验证前不干预导航,
        # 纯透传 move_base 速度)。验证符号后 launch 传 true 再启用。
        self.enable_line = rospy.get_param('~enable_line_follow', False)

        self._mv = None
        self._dev = 0.0
        self._valid = False
        self._last_out_t = 0.0

        rospy.Subscriber('/move_base/cmd_vel', Twist, self._cb_mv, queue_size=1)
        rospy.Subscriber('/visual_line/deviation', Float32, self._cb_dev,
                         queue_size=1)
        rospy.Subscriber('/visual_line/valid', Bool, self._cb_valid,
                         queue_size=1)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        rospy.Timer(rospy.Duration(0.05), self._tick)
        rospy.loginfo('[merge] 就绪: 循线修正=%s (kp=%.2f vy_max=%.2f, move_base/cmd_vel -> cmd_vel)',
                      '开' if self.enable_line else '关', self.kp, self.vy_max)

    def _cb_mv(self, msg):
        self._mv = msg

    def _cb_dev(self, msg):
        self._dev = msg.data

    def _cb_valid(self, msg):
        self._valid = msg.data

    def _tick(self, event):
        """50Hz 转发: move_base 速度 + 循线横向修正。"""
        mv = self._mv
        if mv is None:
            return
        out = Twist()
        out.linear.x = mv.linear.x
        out.linear.y = mv.linear.y
        out.angular.z = mv.angular.z
        if self.enable_line and self._valid and abs(mv.linear.x) > self.min_vx:
            vy = -self.kp * self._dev
            vy = max(-self.vy_max, min(self.vy_max, vy))
            out.linear.y += vy
        self.pub.publish(out)


def main():
    rospy.init_node('cmd_vel_merge', anonymous=False)
    node = CmdVelMerge()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
