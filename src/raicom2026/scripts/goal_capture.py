#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# goal_capture.py —— 捕捉 /move_base_simple/goal 的坐标与朝向，按空格写入文本
# ------------------------------------------------------------
# 订阅 /move_base_simple/goal (geometry_msgs/PoseStamped)，保存最近一帧；
# 在终端里按空格把当前目标点 (x, y, z) + 四元数 + yaw 追加写入文本文件。
#
#   Space    把最近收到的目标点写入文件 (每按一次写一行，带递增序号)
#   H        帮助
#   Ctrl+C   退出
#
# 用法 (必须手动在真实终端跑, roslaunch 起的节点没有 stdin 收不到按键):
#   终端2  rosrun raicom2026 goal_capture.py
#   可加 --topic /your_topic 改订阅话题 (默认 /move_base_simple/goal)
#   可加 --out /path/file.txt 改输出文件 (默认 <包>/data/goals.txt)
# ============================================================
import os
import sys
import time
import select
import termios
import tty
import math

import rospy
from geometry_msgs.msg import PoseStamped

PKG_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DEFAULT_OUT = os.path.join(PKG_DATA, 'goals.txt')


def quat_to_yaw(x, y, z, w):
    """四元数 -> yaw (rad, 范围 ±pi)，x 前 / y 左坐标系。"""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def getch(timeout):
    """非阻塞读一个字节 (timeout 秒), 无输入返回 None。"""
    fd = sys.stdin.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    return os.read(fd, 1)


class GoalCapture:
    def __init__(self, topic='/move_base_simple/goal', out_path=DEFAULT_OUT):
        rospy.init_node('goal_capture', anonymous=True)
        self.out_path = os.path.abspath(out_path)
        self.latest = None            # 最近收到的 PoseStamped
        self.latest_time = 0.0
        self.count = 0
        self.running = True
        self._init_file()
        rospy.Subscriber(topic, PoseStamped, self._on_goal)
        print(f'[goal_capture] 订阅 {topic}\n'
              f'[goal_capture] 输出文件 {self.out_path}')

    # ---------- 文件 ----------
    def _init_file(self):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        if not os.path.exists(self.out_path):
            with open(self.out_path, 'w', encoding='utf-8') as f:
                f.write('# idx  x  y  z  qx  qy  qz  qw  yaw\n')

    def _append(self):
        if self.latest is None:
            print('[goal_capture] 还没收到目标点, 先发一个 /move_base_simple/goal '
                  '再按空格')
            return
        p = self.latest.pose
        yaw = quat_to_yaw(p.orientation.x, p.orientation.y,
                          p.orientation.z, p.orientation.w)
        line = (f'{self.count}  {p.position.x:.4f}  {p.position.y:.4f}  '
                f'{p.position.z:.4f}  {p.orientation.x:.6f}  '
                f'{p.orientation.y:.6f}  {p.orientation.z:.6f}  '
                f'{p.orientation.w:.6f}  {yaw:.4f}\n')
        with open(self.out_path, 'a', encoding='utf-8') as f:
            f.write(line)
        age = time.time() - self.latest_time
        print(f'[goal_capture] #{self.count} 写入 (x={p.position.x:.4f}, '
              f'y={p.position.y:.4f}, yaw={yaw:.4f})  '
              f'(距接收 {age:.1f}s)')
        self.count += 1

    # ---------- 回调 ----------
    def _on_goal(self, msg):
        self.latest = msg
        self.latest_time = time.time()

    # ---------- 主循环 ----------
    def run(self):
        if not sys.stdin.isatty():
            print('[goal_capture] 错误: 没有终端键盘 (stdin 非 tty)。'
                  '本节点必须用 rosrun 在真实终端里运行, 不要被 roslaunch 拉起。',
                  file=sys.stderr)
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            print('[goal_capture] 按空格写入当前目标点, Ctrl+C 退出')
            while self.running and not rospy.is_shutdown():
                ch = getch(0.1)
                if ch is None:
                    continue
                if ch in (b'\x03', b'\x04'):      # Ctrl+C / Ctrl+D
                    print('\n[goal_capture] 退出')
                    break
                if ch == b' ':
                    self._append()
                elif ch in (b'h', b'H'):
                    self._help()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _help(self):
        print('[goal_capture] 空格=写入目标点  H=帮助  Ctrl+C=退出')


if __name__ == '__main__':
    topic = '/move_base_simple/goal'
    out_path = DEFAULT_OUT
    if '--topic' in sys.argv:
        i = sys.argv.index('--topic')
        if i + 1 < len(sys.argv):
            topic = sys.argv[i + 1]
    if '--out' in sys.argv:
        i = sys.argv.index('--out')
        if i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]
    GoalCapture(topic, out_path).run()
