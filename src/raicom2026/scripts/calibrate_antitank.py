#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# calibrate_antitank.py —— 反坦克盘单目测距标定 (focal_px + 横向)
# ------------------------------------------------------------
# 标定目标: 让 project_bbox_mono 的单目测距 d 与真实距离 L 一致。
#   d = disc_m * focal_px / 框宽, 若 focal_px 偏小则 d 系统性偏小,
#   盘的地图坐标整体偏近 -> 点云位置不准。修正 focal_px = 旧值 * L/d。
#
# 用法 (prep 起好后, 车停稳且 AMCL 已定位):
#   1) 盘放在车正前方 L 米 (卷尺量准, 横向尽量居中)
#   2) rosrun raicom2026 calibrate_antitank.py --L 1.0
#   3) 按打印的 focal_px 修正值写回配置
# ============================================================
import os
import sys
import time
import argparse

ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.insert(0, ros_path)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, '/home/mowen/cv_bridge_ws/devel/lib/python3/dist-packages')

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospkg
from raicom2026.yolo_onnx import YOLOv8ONNX
from raicom2026.mine_avoid import AT_CLASSES, project_bbox_mono

PKG = rospkg.RosPack().get_path('raicom2026')
AT_MODEL = os.path.join(PKG, 'data', 'models', 'antitank.onnx')
AT_DISC_M = 0.10


def main():
    parser = argparse.ArgumentParser(description='反坦克盘测距标定')
    parser.add_argument('--L', type=float, required=True, help='盘实际距离(米, 卷尺量)')
    parser.add_argument('--focal', type=float, default=610.0, help='当前 focal_px (默认610)')
    parser.add_argument('--scan', type=float, default=6.0, help='采样时长(秒)')
    args = parser.parse_args(rospy.myargv(sys.argv)[1:])

    rospy.init_node('calibrate_antitank', anonymous=False)
    model = YOLOv8ONNX(AT_MODEL, conf_thres=0.4, iou_thres=0.45, intra_op_num_threads=2)
    br = CvBridge()
    frame = {}

    def cb(msg):
        try:
            frame['img'] = br.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    rospy.Subscriber('/camera/color/image_raw', Image, cb, queue_size=1)

    print('等待 Orbbec 帧 ... (盘应放在车正前方 %.2fm, 横向居中)' % args.L)
    t0 = time.time()
    ds = []
    while time.time() - t0 < args.scan and not rospy.is_shutdown():
        if 'img' not in frame:
            time.sleep(0.05)
            continue
        boxes, scores, ids = model.detect(frame['img'])
        for b, s, c in zip(boxes, scores, ids):
            key = AT_CLASSES.get(int(c))
            if key is None or s < 0.4:
                continue
            pt = project_bbox_mono(b, args.focal, args.focal, 320.0, 240.0,
                                   AT_DISC_M, 5.0)
            if pt is not None:
                ds.append((pt[2], pt[0], float(s)))   # (距离, 横向X, 置信度)
        time.sleep(0.1)

    if not ds:
        print('未检测到盘! 检查: 盘在正前方视野内? antitank.onnx 存在? conf>=0.4?')
        return

    # 按距离排序取中位数 (抗抖动/误检)
    ds.sort(key=lambda x: x[0])
    d, X, s = ds[len(ds) // 2]
    fx_new = args.focal * args.L / d
    print('')
    print('=' * 54)
    print('  盘实际距离 L   = %.3f m' % args.L)
    print('  单目测距 d     = %.3f m (conf=%.2f)' % (d, s))
    print('  横向偏移 X     = %+.3f m (右正)' % X)
    print('  测距误差       = %+.3f m' % (d - args.L))
    print('-' * 54)
    print('  focal_px 修正  : %d -> %.1f' % (args.focal, fx_new))
    print('  写回配置       : ~focal_px: %.1f' % fx_new)
    print('=' * 54)
    if abs(X) > 0.05:
        print('⚠ 横向 X 偏大 (>5cm): 相机光轴不正对盘, 重新摆盘或检查 cam_y 标定')
    else:
        print('横向 X 正常 (相机光轴对正)')
    print('')
    print('标定后: 把 focal_px 写进 detect.launch 的 orbbec_mine_antitank 节点')
    print('  <param name="~focal_px" value="%.1f"/>' % fx_new)


if __name__ == '__main__':
    main()
