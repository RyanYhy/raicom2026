#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# teleop_keyboard.py —— Mecanum 全向底盘键盘遥操作（建图/调试用）
# ------------------------------------------------------------
# 坐标系 (REP-103, 机器人系: x 前, y 左, z 逆时针为正):
#   W / S      前进 / 后退     (linear.x = ±v)
#   A / D      左移 / 右移     (linear.y = ±v)   全向横移
#   Q / E      逆时针 / 顺时针旋转 (angular.z = ±w)
#   X / Space  停车 (立即全零)
#   + / -      加速 / 减速 (调 v、w 倍率, 范围 0.1~3.0)
#   H          帮助
#   Ctrl+C     退出并发布零速
# 安全: 1.5s 无按键自动发布零速（防终端卡死 / 误触跑飞 / 驱动保持最后指令）。
# 按住按键不动: 依赖系统键盘重复 → 持续运动；无重复的环境下 点按 ≈ 运动 1.5s 后自停。
# 用法 (必须手动在真实终端跑, roslaunch 起的节点没有 stdin 收不到按键):
#   终端2  rosrun raicom2026 teleop_keyboard.py     # 建图时与 build_map.launch 配合
#   可加 --topic /your_topic 改发布话题 (默认 /cmd_vel)
# ============================================================
import os
import sys
import time
import select
import threading
import termios
import tty

import rospy
from geometry_msgs.msg import Twist

# ---- 基础速度 (m/s, rad/s), +/− 调倍率 ----
BASE_LINEAR = 0.35
BASE_ANGULAR = 0.7
SPEED_MIN, SPEED_MAX = 0.1, 3.0
AUTO_STOP_SEC = 1.5       # 无按键多久后强制停车
PUB_HZ = 10               # 心跳发布频率

# 按键 -> (linear.x 系数, linear.y 系数, angular.z 系数)
KEYS = {
    'w': (+1.0,  0.0,  0.0),
    's': (-1.0,  0.0,  0.0),
    'a': ( 0.0, +1.0,  0.0),
    'd': ( 0.0, -1.0,  0.0),
    'q': ( 0.0,  0.0, +1.0),
    'e': ( 0.0,  0.0, -1.0),
    # 方向键等价映射
    '\x1b[A': (+1.0,  0.0,  0.0),   # Up
    '\x1b[B': (-1.0,  0.0,  0.0),   # Down
    '\x1b[C': ( 0.0,  0.0, -1.0),   # Right -> 顺时针
    '\x1b[D': ( 0.0,  0.0, +1.0),   # Left  -> 逆时针
}
STOP_KEYS = {'x', ' '}


def getch(timeout):
    """非阻塞读一个字节/转义序列 (timeout 秒), 无输入返回 None。"""
    fd = sys.stdin.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    ch = os.read(fd, 1)
    if ch in (b'\x1b', b'\x00', b'\xe0'):          # 转义 / 功能键前缀: 拼出完整序列
        seq = ch
        for _ in range(2):
            ready, _, _ = select.select([fd], [], [], 0.01)
            if not ready:
                break
            seq += os.read(fd, 1)
        return seq
    return ch


class TeleopKeyboard:
    def __init__(self, topic='/cmd_vel'):
        rospy.init_node('teleop_keyboard', anonymous=True)
        self.pub = rospy.Publisher(topic, Twist, queue_size=1)
        self.lock = threading.Lock()
        self.last_twist = Twist()
        self.last_cmd_time = 0.0
        self.speed = 1.0
        self.running = True
        self._pub_thread = threading.Thread(target=self._heartbeat, daemon=True)

    # ---------- 心跳 + 自动停车 ----------
    def _heartbeat(self):
        rate = rospy.Rate(PUB_HZ)
        while self.running and not rospy.is_shutdown():
            now = time.time()
            # 锁内读取并拷贝字段, 避免 publish 序列化时读到并发改写的对象
            with self.lock:
                twist = self.last_twist
                if now - self.last_cmd_time > AUTO_STOP_SEC:
                    twist = Twist()          # 无按键超时 -> 零速
                out = Twist()
                out.linear.x = twist.linear.x
                out.linear.y = twist.linear.y
                out.angular.z = twist.angular.z
            try:
                self.pub.publish(out)
            except Exception:
                pass
            rate.sleep()

    def set(self, lx=0.0, ly=0.0, az=0.0):
        # 换新对象再整体赋值 (不可变语义), 杜绝被心跳半途读到中间态
        t = Twist()
        t.linear.x = lx
        t.linear.y = ly
        t.angular.z = az
        with self.lock:
            self.last_twist = t
            self.last_cmd_time = time.time()

    def stop(self):
        self.set(0.0, 0.0, 0.0)

    # ---------- 主循环 ----------
    def run(self):
        if not sys.stdin.isatty():
            print('[teleop] 错误: 没有终端键盘 (stdin 非 tty)。'
                  '本节点必须用 rosrun 在真实终端里运行, 不要被 roslaunch 拉起。',
                  file=sys.stderr)
            self.pub.publish(Twist())   # 确保零速
            return
        self._pub_thread.start()
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            self._help()
            while self.running and not rospy.is_shutdown():
                ch = getch(0.1)
                if ch is None:
                    continue
                if ch in (b'\x03', b'\x04'):      # Ctrl+C / Ctrl+D
                    print('\n[teleop] 退出并发布零速')
                    self.stop()
                    break
                s = ch.decode('latin1') if isinstance(ch, bytes) else ch
                k = s.lower() if len(s) == 1 else s   # 只小写单字符, 转义序列(方向键)保原样
                if k in KEYS:
                    lx, ly, az = KEYS[k]
                    self.set(lx * BASE_LINEAR * self.speed,
                             ly * BASE_LINEAR * self.speed,
                             az * BASE_ANGULAR * self.speed)
                    print(f'[teleop] v={BASE_LINEAR*self.speed:.2f} '
                          f'w={BASE_ANGULAR*self.speed:.2f} 键={k!r}')
                elif k in STOP_KEYS:
                    self.stop()
                    print('[teleop] STOP')
                elif k == '+':
                    self.speed = min(SPEED_MAX, self.speed + 0.1)
                    print(f'[teleop] 倍率 -> {self.speed:.1f}')
                elif k == '-':
                    self.speed = max(SPEED_MIN, self.speed - 0.1)
                    print(f'[teleop] 倍率 -> {self.speed:.1f}')
                elif k == 'h':
                    self._help()
        except KeyboardInterrupt:
            self.stop()
            print('\n[teleop] 退出并发布零速')
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self.stop()
            # 同步发一次零速, 不依赖心跳线程 (防止此刻心跳在 sleep 错过导致底盘保持最后指令)
            try:
                self.pub.publish(Twist())
            except Exception:
                pass
            time.sleep(0.2)

    def _help(self):
        print('''\
[teleop] Mecanum 全向遥操作
  W/S  前后   A/D  左右横移   Q/E  原地旋转   X/Space 停车
  +/-  速度倍率   H 帮助   Ctrl+C 退出
  安全: 无按键 1.5s 自动停车 (当前倍率 %.1f, v=%.2f w=%.2f)''' % (
            self.speed, BASE_LINEAR * self.speed, BASE_ANGULAR * self.speed))


if __name__ == '__main__':
    topic = '/cmd_vel'
    if '--topic' in sys.argv:
        i = sys.argv.index('--topic')
        if i + 1 < len(sys.argv):          # 防 --topic 在末尾时 IndexError
            topic = sys.argv[i + 1]
    TeleopKeyboard(topic).run()
