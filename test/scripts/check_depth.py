#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# check_depth.py —— 反坦克盘深度值验证 (判断深度测距是否可行)
# ------------------------------------------------------------
# 前置: 已开启 depth_registration (nav_competition_2026.launch 已开),
#       盘放车正前方。
# 用法: rosrun raicom2026 check_depth.py
# 输出: 盘框中心/底边在深度图里的值, 判断:
#   - 深度值稳定有效(0.1~3m 且接近实际距离) -> 反光不严重, 可用深度测距
#   - 深度值 0/NaN 居多 -> 盘反光导致孔洞, 只能继续单目测距
# ============================================================
import os
import sys
import time

ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.insert(0, ros_path)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, '/home/mowen/cv_bridge_ws/devel/lib/python3/dist-packages')

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospkg
from raicom2026.yolo_onnx import YOLOv8ONNX

PKG = rospkg.RosPack().get_path('raicom2026')
AT_MODEL = os.path.join(PKG, 'data', 'models', 'antitank.onnx')


def main():
    rospy.init_node('check_depth', anonymous=False)
    model = YOLOv8ONNX(AT_MODEL, conf_thres=0.4, iou_thres=0.45, intra_op_num_threads=2)
    br = CvBridge()
    rgb = {}
    depth = {}

    def cb_rgb(msg):
        try:
            rgb['img'] = br.imgmsg_to_cv2(msg, 'bgr8')
            rgb['h'], rgb['w'] = rgb['img'].shape[:2]
        except Exception:
            pass

    def cb_depth(msg):
        try:
            enc = msg.encoding or ''
            if '16UC1' in enc or 'mono16' in enc or 'Y11' in enc:
                depth['img'] = br.imgmsg_to_cv2(msg, '16UC1').astype(np.float32) / 1000.0
            elif '32FC1' in enc:
                depth['img'] = br.imgmsg_to_cv2(msg, '32FC1')
            else:
                return
            depth['h'], depth['w'] = depth['img'].shape[:2]
        except Exception as e:
            rospy.logwarn('depth convert fail: %s', e)

    rospy.Subscriber('/camera/color/image_raw', Image, cb_rgb, queue_size=1)
    rospy.Subscriber('/camera/depth/image_raw', Image, cb_depth, queue_size=1)

    print('等待彩色/深度帧 ... (盘放车正前方)')
    print('彩色 %dx%d, 深度 %dx%d 会在首帧打印' % (640, 480, 640, 400))
    t0 = time.time()
    n = 0
    while time.time() - t0 < 30 and not rospy.is_shutdown():
        if 'img' not in rgb or 'img' not in depth:
            time.sleep(0.05)
            continue
        if n == 0:
            print('彩色分辨率 %dx%d, 深度分辨率 %dx%d'
                  % (rgb['w'], rgb['h'], depth['w'], depth['h']))
        boxes, scores, ids = model.detect(rgb['img'])
        if not boxes:
            time.sleep(0.05)
            continue
        # 取置信度最高的盘
        best = max(zip(boxes, scores), key=lambda x: x[1])
        b, s = best
        left, top, w, h = [int(v) for v in b]
        cx = left + w // 2
        cy = top + h // 2
        # 彩色坐标 -> 深度坐标 (配准后应 1:1; 分辨率不同按比例缩放兜底)
        sx = depth['w'] / float(rgb['w'])
        sy = depth['h'] / float(rgb['h'])
        dx = int(np.clip(cx * sx, 0, depth['w'] - 1))
        dy = int(np.clip(cy * sy, 0, depth['h'] - 1))
        # 5x5 patch 中位数 (抗反光单点孔洞)
        x0, x1 = max(0, dx - 2), min(depth['w'], dx + 3)
        y0, y1 = max(0, dy - 2), min(depth['h'], dy + 3)
        patch = depth['img'][y0:y1, x0:x1]
        valid = patch[(patch > 0.05) & np.isfinite(patch)]
        z_med = float(np.median(valid)) if valid.size else 0.0
        z_cen = float(depth['img'][dy, dx])
        n += 1
        if n % 3 == 1:
            print('框中心(%d,%d) -> 深度(%d,%d) 中心=%.3fm 5x5中位=%.3fm (有效%d/%d) conf=%.2f'
                  % (cx, cy, dx, dy, z_cen, z_med, valid.size, patch.size, s))
        time.sleep(0.1)

    if n == 0:
        print('30s 内未检测到盘! 检查盘位置 + antitank.onnx')
    else:
        print('')
        print('=' * 54)
        print('判断: 5x5中位 稳定在 0.1~3m 且接近实际距离 -> 深度可用')
        print('      大部分为 0.000 或 NaN -> 盘反光, 只能继续单目')
        print('=' * 54)


if __name__ == '__main__':
    main()
