#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# camera_bridge.py —— 云台 webcam 相机桥节点 (替代 usb_cam_node)
# ------------------------------------------------------------
# 背景 (2026-08-12 实车排查):
#   usb_cam_node (MJPG 848x480) 在该 Realtek/Sonix 相机上大量
#   "Error while decoding frame", 实际发布画面黑白撕裂不可用;
#   直读 cv2.VideoCapture(0) (YUYV 640x480 相机默认) 画面正常,
#   模型识别恢复正常 (person_best2 三偶全检出)。
# 职责:
#   直读 /dev/video0 (YUYV 640x480) -> cv_bridge -> /usb_cam/image_raw
#   (frame_id=usb_cam, 与旧 usb_cam 输出话题/帧名完全一致, 下游零改动)
# 关键设计:
#   - 失败重开重试: 该相机首次 STREAMON 常报 Protocol error,
#     重新打开设备后即恢复 (v4l2/cv2 实测模式)。
#   - 无帧时不疯狂重开 (限频), 保持话题 hz。
# 用法:
#   roslaunch raicom2026 usb_cam_mjpeg.launch   (launch 内已指向本节点)
#   rosrun raicom2026 camera_bridge.py
# ⚠️ 分辨率由 848x480 -> 640x480: param/params.yaml 的 ppd_h/ppd_v
#   需按比例换算 (12.11 * 640/848 ≈ 9.14) 或重新标定, 否则打靶换算偏 1.325 倍。
# ============================================================
import os
import sys
import time

# ---- 环境兼容补丁 (必须最前: melodic 的 py2 rospy) ----
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.insert(0, ros_path)

# cv_bridge 必须 python3 版并提到最前
sys.path.insert(0, '/home/mowen/cv_bridge_ws/devel/lib/python3/dist-packages')

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

IMAGE_TOPIC = '/usb_cam/image_raw'
FRAME_ID = 'usb_cam'
TARGET_FPS = 30.0
REOPEN_MIN_INTERVAL = 2.0    # 重开间隔下限 (秒), 防止无帧时疯狂重开


class CameraBridge:
    def __init__(self):
        self.br = CvBridge()
        self.pub = rospy.Publisher(IMAGE_TOPIC, Image, queue_size=1)
        self.cap = None
        self._last_reopen = 0.0
        self.dev_idx = self._find_usb_camera()
        if self.dev_idx is None:
            rospy.logfatal('[cam] 未找到 USB 摄像头 (遍历 /dev/video* 无匹配), 退出')
            sys.exit(1)
        self._open()
        if self.cap is None or not self.cap.isOpened():
            rospy.logfatal('[cam] 无法打开 /dev/video%d, 退出', self.dev_idx)
            sys.exit(1)
        rospy.loginfo('[cam] 相机桥就绪: 直读 /dev/video%d -> %s', self.dev_idx, IMAGE_TOPIC)
        rospy.Timer(rospy.Duration(1.0 / TARGET_FPS), self._tick)

    def _find_usb_camera(self):
        """自动探测 USB 摄像头 (2026-08-15 v2): Orbbec 启动后会抢占 /dev/video0。
        只排除 Orbbec(obsensor/orbbec) 设备, 第一个非 Orbbec 的 video 就是摄像头。
        ⚠️ 注意: 实车的 USB 摄像头(Realtek 0bda:3031)产品名就叫
        "Integrated Webcam", 不能按 integrated 关键字排除(否则永远找不到)。"""
        best = None
        for i in range(10):
            path = '/dev/video%d' % i
            if not os.path.exists(path):
                continue
            name = ''
            try:
                with open('/sys/class/video4linux/video%d/name' % i) as f:
                    name = f.read().strip().lower()
            except Exception:
                pass
            if 'obsensor' in name or 'orbbec' in name:
                continue
            best = i
            rospy.loginfo('[cam] 选定摄像头: video%d (名=%s)', i, name)
            break
        return best

    def _open(self):
        """打开相机 (默认参数 640x480 YUYV); 失败/重开前先释放旧句柄。"""
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            self.cap = cv2.VideoCapture(self.dev_idx)
            if self.cap.isOpened():
                rospy.loginfo('[cam] 相机打开成功 (%dx%d)',
                              int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        except Exception as e:
            rospy.logwarn_throttle(5.0, '[cam] 打开相机失败: %s', e)
            self.cap = None

    def _tick(self, event):
        if self.cap is None or not self.cap.isOpened():
            if time.time() - self._last_reopen >= REOPEN_MIN_INTERVAL:
                self._last_reopen = time.time()
                self._open()
            return
        try:
            ok, img = self.cap.read()
        except Exception as e:
            ok = False
            rospy.logwarn_throttle(5.0, '[cam] read 异常: %s', e)
        if not ok or img is None:
            # 首次 STREAMON 常报协议错误: 关闭重开即恢复 (限频)
            if time.time() - self._last_reopen >= REOPEN_MIN_INTERVAL:
                self._last_reopen = time.time()
                rospy.logwarn_throttle(3.0, '[cam] 读帧失败, 重开相机')
                self._open()
            return
        try:
            msg = self.br.cv2_to_imgmsg(img, 'bgr8')
            msg.header.frame_id = FRAME_ID
            msg.header.stamp = rospy.Time.now()
            self.pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, '[cam] 发布失败: %s', e)

    def shutdown(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


def _sig_to_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    import signal
    rospy.init_node('camera_bridge', anonymous=False)
    signal.signal(signal.SIGTERM, _sig_to_interrupt)
    signal.signal(signal.SIGINT, _sig_to_interrupt)
    node = CameraBridge()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rospy.loginfo('[cam] 退出')


if __name__ == '__main__':
    main()
