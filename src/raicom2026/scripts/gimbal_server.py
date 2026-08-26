#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# gimbal_server.py —— RAICOM2026 智能侦察 云台+水弹枪 串口服务
# ------------------------------------------------------------
# 职责: 独占 /dev/gun 串口, 提供舵机/开火的服务化接口。
#       __不允许__ 任何其他进程再打开 /dev/gun (nav_competition.py 的
#       串口块必须停用, 改走本节点服务)。
#
# 协议 (实测, 见 cam_results/外设使用注意事项.md §4.5):
#   舵机1(水平):  AA BB 01 <angle> 0D 0A   angle: 0-180, 中心 90
#   舵机2(俯仰):  AA BB 02 <angle> 0D 0A   angle: 70-110, 中心 90
#   水弹枪开火:    AA BB 03 01 0D 0A
#   水弹枪停止:    AA BB 03 00 0D 0A
#   0.05s 点射有效; 冷却 3s (散热)
#
# 服务:
#   /gun/aim  (GunAim)  : 设置 pan(舵机1)/tilt(舵机2) 绝对角度
#   /gun/fire (GunFire) : 水弹枪点射 (burst_ms 默认 50)
#   /gun/home (Trigger) : 云台归位 (pan=90, tilt=90)
#   /gun/get  (GunState): 查询当前角度
#
# 运行: 在 yolov8 conda 环境执行
#   rosrun raicom2026 gimbal_server.py
# 或
#   roslaunch raicom2026 gimbal.launch
# ============================================================
import sys
import os
import time
import atexit

# ---- 环境兼容补丁 (必须放在最前面) ----
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.insert(0, ros_path)

import serial
import rospy
from std_srvs.srv import Trigger, TriggerResponse

# 本包自定义服务
from raicom2026.srv import GunAim, GunAimResponse, GunFire, GunFireResponse


class GimbalServer:
    SERVO1_MIN = 0       # 舵机1 (pan) 角度范围
    SERVO1_MAX = 180
    SERVO1_CENTER = 90
    SERVO2_MIN = 70      # 舵机2 (tilt) 角度范围
    SERVO2_MAX = 110
    SERVO2_CENTER = 90

    SERVO_CMD = 0x01
    TILT_CMD = 0x02
    FIRE_ON = 0x03
    SETTLE_TIME = 0.1    # 每个指令写入后的稳定时间 (秒)
    FIRE_COOLDOWN = 1.5  # 开火冷却 (2026-08-16: 3.0->1.5, 快节奏打靶)
    DEFAULT_BURST = 0.05  # 点射时长 (秒, 实测 0.05s 有效)

    def __init__(self, port='/dev/gun', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.servo1 = self.SERVO1_CENTER
        self.servo2 = self.SERVO2_CENTER
        self.last_fire_time = 0.0
        self._last_reconnect_try = 0.0
        self.ser = None

        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.5)
            rospy.loginfo(f"[gimbal] 串口打开成功 {port} @ {baudrate}")
        except Exception as e:
            rospy.logfatal(f"[gimbal] 串口打开失败 {port}: {e}")
            rospy.logwarn("[gimbal] 云台服务将以'串口不可用'模式继续(所有命令返回 False)")

        # 上电归位
        self.send_servo(self.SERVO_CMD, self.SERVO1_CENTER)
        self.send_servo(self.TILT_CMD, self.SERVO2_CENTER)
        time.sleep(0.3)
        rospy.loginfo(f"[gimbal] 初始化完成 pan={self.servo1} tilt={self.servo2}")

        atexit.register(self.shutdown)

    def shutdown(self):
        """退出前自动归位云台到安全位置, 再释放串口。
        无论节点如何退出(kill/崩溃/正常关), 舵机都回到 pan=90 tilt=90,
        避免长时间顶在限位角发热, 也避免下次上电时舵机不在已知位置。"""
        if self.ser is not None and self.ser.is_open:
            try:
                self.send_servo(self.SERVO_CMD, self.SERVO1_CENTER)
                self.send_servo(self.TILT_CMD, self.SERVO2_CENTER)
                time.sleep(0.3)
                rospy.loginfo(f"[gimbal] 退出前云台归位完成 pan={self.SERVO1_CENTER} tilt={self.SERVO2_CENTER}")
            except Exception as e:
                rospy.logwarn(f"[gimbal] 退出归位失败: {e}")
            try:
                self.ser.close()
                rospy.loginfo("[gimbal] 串口已关闭")
            except Exception:
                pass

    # ---------- 底层串口 ----------
    def _try_reconnect(self):
        """串口掉线自动重连 (2026-08-14): USB 松动/瞬时掉电导致 /dev/gun 消失,
        设备重枚举后自动重开串口并归位, 无需人工重启节点。限频 3s。"""
        now = time.time()
        if now - self._last_reconnect_try < 3.0:
            return False
        self._last_reconnect_try = now
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        if not os.path.exists(self.port):
            return False
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
            rospy.loginfo("[gimbal] 串口重连成功 %s" % self.port)
            # 重连后舵机状态未知, 归位
            self.send_servo(self.SERVO_CMD, self.SERVO1_CENTER)
            self.send_servo(self.TILT_CMD, self.SERVO2_CENTER)
            return True
        except Exception as e:
            rospy.logwarn_throttle(10.0, "[gimbal] 串口重连失败: %s" % e)
            return False

    def send_cmd(self, data):
        """写一条指令并等待稳定。返回 True/False。
        掉线时自动尝试重连并重写一次 (2026-08-14)。"""
        if self.ser is None or not self.ser.is_open:
            if not self._try_reconnect():
                return False
        try:
            self.ser.write(data)
            time.sleep(self.SETTLE_TIME)
            return True
        except Exception as e:
            rospy.logerr(f"[gimbal] 串口写入失败: {e}")
            if self._try_reconnect():
                try:
                    self.ser.write(data)
                    time.sleep(self.SETTLE_TIME)
                    return True
                except Exception:
                    pass
            return False

    def send_servo(self, cmd, angle):
        return self.send_cmd(bytes([0xAA, 0xBB, cmd, int(angle), 0x0D, 0x0A]))

    # ---------- 动作 ----------
    def set_pan(self, angle):
        """设置水平舵机绝对角度 (0-180)。返回 (ok, msg)。"""
        angle = int(round(angle))
        angle = max(self.SERVO1_MIN, min(self.SERVO1_MAX, angle))
        if self.send_servo(self.SERVO_CMD, angle):
            self.servo1 = angle
            return True, f"pan={angle}"
        return False, "serial write failed"

    def set_tilt(self, angle):
        """设置俯仰舵机绝对角度 (70-110)。返回 (ok, msg)。"""
        angle = int(round(angle))
        angle = max(self.SERVO2_MIN, min(self.SERVO2_MAX, angle))
        if self.send_servo(self.TILT_CMD, angle):
            self.servo2 = angle
            return True, f"tilt={angle}"
        return False, "serial write failed"

    def _fire_off(self, off_cmd, retries=10):
        """关枪指令重试 (关闭写失败 = 水弹枪可能持续喷射, 必须尽量保证发出去)。
        2026-08-16: 3->10 次 + 间隔加长 + 串口重连重发 (开火可能干扰串口)。"""
        for _ in range(retries):
            if self.send_cmd(off_cmd):
                return True
            time.sleep(0.1)
        # 串口重连后重发 (开火电流干扰可能导致串口瞬断)
        self._try_reconnect()
        for _ in range(5):
            if self.send_cmd(off_cmd):
                return True
            time.sleep(0.1)
        return False

    def fire(self, burst_ms=50):
        """水弹枪点射。默认 0.05s。遵守冷却时间。"""
        now = time.time()
        if now - self.last_fire_time < self.FIRE_COOLDOWN:
            wait = self.FIRE_COOLDOWN - (now - self.last_fire_time)
            rospy.logwarn(f"[gimbal] 冷却中, 还需 {wait:.1f}s")
            return False, f"cooldown {wait:.1f}s"

        on_cmd = bytes([0xAA, 0xBB, self.FIRE_ON, 0x01, 0x0D, 0x0A])
        off_cmd = bytes([0xAA, 0xBB, self.FIRE_ON, 0x00, 0x0D, 0x0A])
        if not self.send_cmd(on_cmd):
            return False, "serial write failed"

        burst = burst_ms / 1000.0 if burst_ms > 0 else self.DEFAULT_BURST
        time.sleep(burst)

        self.last_fire_time = time.time()   # 无论关枪成败立即进入冷却, 防连射
        if not self._fire_off(off_cmd):
            rospy.logfatal("[gimbal] 关枪指令写失败! 水弹枪可能仍在发射, 尝试归位")
            try:
                self.home()
            except Exception:
                pass
            # 2026-08-16: 归位后最后一次强制关枪 (绝不能留枪持续喷射)
            try:
                if self._fire_off(off_cmd, retries=15):
                    rospy.loginfo("[gimbal] 补救关枪成功")
            except Exception:
                pass
            return False, "fire-off failed (gun may be firing!)"

        return True, f"fired burst={burst_ms}ms"

    def home(self):
        ok1, m1 = self.set_pan(self.SERVO1_CENTER)
        ok2, m2 = self.set_tilt(self.SERVO2_CENTER)
        return ok1 and ok2, f"{m1} {m2}"

    # ---------- 服务回调 ----------
    def cb_aim(self, req):
        rospy.loginfo(f"[gimbal] /gun/aim pan={req.pan} tilt={req.tilt}")
        ok1, m1 = self.set_pan(req.pan)
        ok2, m2 = self.set_tilt(req.tilt)
        ok = ok1 and ok2
        msg = f"{m1}; {m2}"
        return GunAimResponse(success=ok, message=msg)

    def cb_fire(self, req):
        rospy.loginfo(f"[gimbal] /gun/fire burst={req.burst_ms}")
        ok, msg = self.fire(req.burst_ms)
        return GunFireResponse(success=ok, message=msg)

    def cb_home(self, req):
        rospy.loginfo("[gimbal] /gun/home")
        ok, msg = self.home()
        return TriggerResponse(success=ok, message=msg)

    def cb_get(self, req):
        resp = TriggerResponse()
        resp.success = True
        resp.message = f"pan={self.servo1} tilt={self.servo2}"
        return resp


def main():
    rospy.init_node('gimbal_server', anonymous=False)

    port = rospy.get_param('~port', '/dev/gun')
    baudrate = rospy.get_param('~baudrate', 115200)

    server = GimbalServer(port=port, baudrate=baudrate)

    rospy.Service('/gun/aim', GunAim, server.cb_aim)
    rospy.Service('/gun/fire', GunFire, server.cb_fire)
    rospy.Service('/gun/home', Trigger, server.cb_home)
    rospy.Service('/gun/get', Trigger, server.cb_get)

    rospy.loginfo("[gimbal] 云台服务就绪: /gun/aim /gun/fire /gun/home /gun/get")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
