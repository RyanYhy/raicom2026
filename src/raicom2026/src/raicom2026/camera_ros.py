#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# raicom2026/camera_ros.py —— 订阅式相机封装(自动拉起相机/roscore)
# ------------------------------------------------------------
# 从 collect.py 的 orbbec 自动打开模式提炼, webcam / orbbec 共用:
#   1) 若 ROS master 未运行 → 自动拉起 roscore
#   2) 订阅 image 话题; 2s 内无帧 → 自动 roslaunch 指定的相机 launch
#      (webcam: 本包 usb_cam_mjpeg.launch; orbbec: dabai_dcw2.launch)
#   3) 退出时自动关掉本封装拉起的 roscore / roslaunch (不影响已存在的节点)
# 依赖: cv_bridge 必须用 python3 构建版 (导入前把
# /home/mowen/cv_bridge_ws/devel/lib/python3/dist-packages 提到 sys.path 最前)。
# 注意: 调用前必须先 rospy.init_node() (本封装不 init)。
# ============================================================
import os
import sys
import time
import socket
import signal
import urllib.parse
import subprocess

# ---- 环境兼容补丁 ----
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.insert(0, ros_path)

import rospy
from sensor_msgs.msg import Image

# cv_bridge 必须 python3 版且提到路径最前 (否则解析到 python2 cv_bridge,
# boost 模块 ABI 不兼容, 报 PyInit_cv_bridge_boost, 转换必失败)
sys.path.insert(0, '/home/mowen/cv_bridge_ws/devel/lib/python3/dist-packages')
from cv_bridge import CvBridge


_AUTO_MASTER_PROC = None   # ensure_master 拉起的 roscore, 由 RosCamera.shutdown 回收


def _master_alive():
    """健壮解析 ROS_MASTER_URI (容忍尾斜杠/IPv6), 判断 master 是否在线。"""
    try:
        uri = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
        parsed = urllib.parse.urlsplit(uri)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 11311
        s = socket.create_connection((host, int(port)), timeout=1.0)
        s.close()
        return True
    except Exception:
        return False


def ensure_master():
    """若无 ROS master 先拉起 roscore。必须在 rospy.init_node() 之前调用:
    init_node 在无 master 时会直接抛 ROSInitException, RosCamera.__init__ 里
    的自动拉起根本来不及执行 (评审发现)。
    返回 True 表示 master 可用。拉起的 roscore 记入 _AUTO_MASTER_PROC,
    由 RosCamera.shutdown() 统一回收。"""
    global _AUTO_MASTER_PROC
    if _master_alive():
        return True
    rospy.logwarn('[camera_ros] ROS master 未运行, 自动拉起 roscore ...')
    try:
        _AUTO_MASTER_PROC = subprocess.Popen(
            ['roscore'], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 10 and not _master_alive():
            time.sleep(0.2)
    except Exception as e:
        rospy.logwarn('[camera_ros] 自动拉起 roscore 失败: %s', e)
    return _master_alive()


class RosCamera:
    """订阅 image 话题的相机; 无帧时自动拉起 launch, 退出自动回收。

    get_frame() 返回最新 BGR ndarray 或 None; has_frame() 判断是否出帧。
    """

    def __init__(self, topic, launch_cmd, wait=20, log='/tmp/recon_camera.log',
                 fresh_secs=2.0):
        self.topic = topic
        self.launch_cmd = launch_cmd
        self.wait = wait
        self.log = log
        self.fresh_secs = fresh_secs          # 超过此时间无新帧视为相机死亡
        self.w = self.h = 0
        self._frame = None
        self._seq = 0
        self._last_cb_t = 0.0
        self._proc = None
        self._roscore_proc = None
        self._bridge = CvBridge()

        if not _master_alive():
            self._start_roscore()

        rospy.Subscriber(topic, Image, self._cb, queue_size=1)

        # 先等已有的相机节点发帧 (2s)
        t0 = time.time()
        while time.time() - t0 < 2 and not self.has_frame() and not rospy.is_shutdown():
            time.sleep(0.05)
        if not self.has_frame():
            self._launch()
            t0 = time.time()
            while (time.time() - t0 < self.wait and not self.has_frame()
                   and not rospy.is_shutdown()):
                time.sleep(0.1)
                if self._proc and self._proc.poll() is not None:
                    break

        if self.has_frame():
            rospy.loginfo('[camera_ros] %s 出帧 %dx%d', self.topic, self.h, self.w)
        else:
            rospy.logwarn('[camera_ros] %s 无帧(设备未插/被占用/供电不足?), 日志: %s',
                          self.topic, self.log)

    # ---------- 取帧 ----------
    def _cb(self, msg):
        try:
            self._frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._seq += 1
            self._last_cb_t = time.time()
            if not self.w:
                self.h, self.w = self._frame.shape[:2]
        except Exception:
            pass

    def _fresh(self):
        """相机是否还在出帧 (超过 fresh_secs 无新帧视为死亡/被拔)。"""
        return (self._last_cb_t > 0 and
                time.time() - self._last_cb_t <= self.fresh_secs)

    def has_frame(self):
        return self._frame is not None and self._fresh()

    def get_frame(self):
        """取最新帧 (引用, 不拷贝); 无帧或帧已过期(相机死)返回 None。
        避免相机死后一直拿陈旧帧当活数据继续报数。"""
        if not self._fresh():
            return None
        return self._frame

    def get_frame_seq(self):
        """返回 (frame, seq)。seq 单调递增, 用于避免把同一物理帧重复处理
        (相机 ~10fps 而消费循环 ~30ms, 直接 get_frame 会 3 次处理同一帧,
        导致 '连续 N 帧确认' 被单帧击穿)。过期返回 (None, -1)。"""
        if not self._fresh():
            return None, -1
        return self._frame, self._seq

    def clear_frame(self):
        """消费一帧后清掉, 防止下一轮扫到旧帧。"""
        self._frame = None

    # ---------- 自动拉起 ----------
    def _start_roscore(self):
        rospy.logwarn('[camera_ros] ROS master 未运行, 自动拉起 roscore ...')
        try:
            self._roscore_proc = subprocess.Popen(
                ['roscore'], start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t0 = time.time()
            while time.time() - t0 < 10 and not _master_alive():
                time.sleep(0.2)
        except Exception as e:
            rospy.logwarn('[camera_ros] 自动拉起 roscore 失败: %s', e)

    def _launch(self):
        rospy.loginfo('[camera_ros] 自动启动: %s', self.launch_cmd)
        try:
            env = os.environ.copy()
            # orbbec_camera 已在 1raicom_ws 内；请先 source scripts/setup_env.sh
            logf = open(self.log, 'w')
            self._proc = subprocess.Popen(
                self.launch_cmd.split(), start_new_session=True,
                stdout=logf, stderr=subprocess.STDOUT, env=env)
        except Exception as e:
            rospy.logwarn('[camera_ros] 自动启动失败: %s', e)

    def shutdown(self):
        """关掉本封装拉起的 roscore / roslaunch (不影响已存在的节点)。"""
        global _AUTO_MASTER_PROC
        procs = [('roslaunch', self._proc), ('roscore', self._roscore_proc)]
        if _AUTO_MASTER_PROC is not None:
            procs.append(('roscore(auto)', _AUTO_MASTER_PROC))
            _AUTO_MASTER_PROC = None
        for name, proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    proc.wait(timeout=5)
                    rospy.loginfo('[camera_ros] 已停止本封装拉起的 %s', name)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
