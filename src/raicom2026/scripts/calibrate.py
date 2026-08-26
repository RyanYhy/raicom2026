#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# calibrate.py —— RAICOM2026 标定工具
# ------------------------------------------------------------
# 功能1: 相机内参标定 (棋盘格) —— 输出 K 矩阵/畸变系数
# 功能2: 云台 px→度 (ppd_h/ppd_v) + 符号因子 —— 输出 params.yaml
# 功能3: 反坦克盘深度验证 (Orbbec, 可选)
#
# 用法 (相机标定):
#   1) 打印 9x6 棋盘格 (格子边长 ~15mm), 对着 webcam/Orbbec 拍 25-30 张
#   2) rosrun raicom2026 calibrate.py --calib --device video0 \
#        --cols 9 --rows 6 --square 0.015 --out param/camera_intrinsics.npz
# 用法 (云台 px->度):
#   1) 云台归位 90 度, 在画面正中放一个特征物
#   2) 记下特征物像素 x0, 让云台转已知角度 dpan
#   3) 再记特征物像素 x1 -> ppd_h = |x1-x0| / dpan
#   rosrun raicom2026 calibrate.py --ppd --device video0 \
#        --dpan 10 --out param/params.yaml
# ============================================================
import os
import sys
import glob
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from raicom2026.yolo_onnx import find_chinese_font


def do_calib(device, cols, rows, square, out, cap_scale=(848, 480)):
    """棋盘格相机内参标定。"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    objpoints, imgpoints = [], []
    cap = cv2.VideoCapture(device if device.isdigit() else device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("无法打开相机", file=sys.stderr)
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_scale[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_scale[1])

    print("按 空格 采集当前帧(需看到完整棋盘格), Q 完成")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, (cols, rows), corners, found)
        cv2.putText(disp, f"collected: {len(objpoints)}/30  SPACE采集 Q完成",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('calib', disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' ') and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"  +1 (共 {len(objpoints)})")

    cv2.destroyAllWindows()
    cap.release()
    if len(objpoints) < 10:
        print("有效棋盘格太少 (<10), 标定失败", file=sys.stderr)
        return
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez(out, mtx=mtx, dist=dist)
    print(f"\n相机内参 K:\n{mtx}")
    print(f"畸变: {dist.ravel()}")
    print(f"重投影误差: {ret:.4f} (越小越好, <1 佳)")
    print(f"已保存 -> {out}")


def do_ppd(device, dpan, out):
    """云台 px->度 标定。要求: 云台已经归位 90°, 画面中心有特征物。
    流程: 记录特征物当前像素 -> 调 /gun/aim 转 dpan 度 -> 再次记录像素。
    计算 ppd_h = |x1-x0|/dpan。"""
    # 只做像素采集 + 计算说明, 实际角度由 gimbal_server 服务提供
    cap = cv2.VideoCapture(device if device.isdigit() else device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("无法打开相机", file=sys.stderr)
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pos = []
    print("步骤: 1) 在画面中点击特征物中心 2) 调云台转 %d 度后再次点击 3) 回车计算" % dpan)
    clicks = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            print(f"  记录点: ({x}, {y})")

    cv2.namedWindow('ppd')
    cv2.setMouseCallback('ppd', mouse_cb)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        for (x, y) in clicks:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"clicks: {len(clicks)}/2  回车计算", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('ppd', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == 13 and len(clicks) == 2:
            dx = abs(clicks[1][0] - clicks[0][0])
            ppd_h = dx / dpan
            print(f"\n像素位移 dx={dx}px, 云台角 {dpan}°")
            print(f"ppd_h = {ppd_h:.3f} px/度  (HFOV ≈ {848/ppd_h:.1f}°)")
            print(f"写入 params.yaml 建议值 (需配合 sign 因子)")
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            with open(out, 'w') as f:
                f.write(f"# 云台标定结果\nppd_h: {ppd_h:.3f}\nppd_v: {ppd_h:.3f}  # 假定同比例, 应单独标定\n")
            print(f"已保存 -> {out}")
            break
        if len(clicks) > 2:
            clicks = clicks[-2:]

    cv2.destroyAllWindows()
    cap.release()


def main():
    parser = argparse.ArgumentParser(description='标定工具')
    sub = parser.add_subparsers(dest='cmd')

    p_calib = sub.add_parser('calib', help='棋盘格相机标定')
    p_calib.add_argument('--device', default='0')
    p_calib.add_argument('--cols', type=int, default=9)
    p_calib.add_argument('--rows', type=int, default=6)
    p_calib.add_argument('--square', type=float, default=0.015, help='格子边长(米)')
    p_calib.add_argument('--out', default='data/param/camera_intrinsics.npz')

    p_ppd = sub.add_parser('ppd', help='云台 px->度')
    p_ppd.add_argument('--device', default='0')
    p_ppd.add_argument('--dpan', type=float, default=10.0, help='云台转角(度)')
    p_ppd.add_argument('--out', default='param/params.yaml')

    args = parser.parse_args()
    if args.cmd == 'calib':
        do_calib(args.device, args.cols, args.rows, args.square, args.out)
    elif args.cmd == 'ppd':
        do_ppd(args.device, args.dpan, args.out)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
