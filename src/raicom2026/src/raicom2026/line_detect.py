# -*- coding: utf-8 -*-
# ============================================================
# raicom2026/line_detect.py —— 车道双黄线检测纯逻辑 (2026-08-14 v2)
# ------------------------------------------------------------
# 比赛场地: 车道两侧各一条黄色边线 (~5cm 宽, 间距约 480mm),
# 小车必须在两条黄线中间行驶, 不能踩线。
# 相机: Orbbec DaBai DCW2, 640x480, 高 0.2m 平视。
# 只依赖 numpy+cv2, 不 import rospy (离线可测)。
#
# 算法 (质心二分, 对双线对称车道天然稳健):
#   1) 画面下部 ROI (车前方 ~0.5-1.5m 地面) 转 HSV
#   2) 黄色掩膜 + 开运算去噪
#   3) 全体黄像素质心 cx_all -> 按 cx_all 分左右两组
#      左组质心=左线, 右组质心=右线
#   4) 车道中心 = 两线质心均值; 偏差 = 车道中心 - 画面中心
#   5) 校验: 两线间距在合理范围 (车道 480mm @0.5~1.5m -> ~150-600px),
#      否则视为误检返回 None
# ============================================================
import numpy as np
import cv2

# HSV 黄色范围 (胶带黄; H 18-35, S/V 下限防阴影灰)
YELLOW_LOW = np.array([18, 90, 100], dtype=np.uint8)
YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

# 检测有效性: 每侧线最少像素数 (2026-08-15: 30->5, 窄道板子可能挡住部分黄线)
MIN_LINE_PX = 5
# 两线间距合理范围 (px): 车道 480mm, 距离 0.5m->~590px, 1.5m->~195px
LANE_PX_MIN = 120.0
LANE_PX_MAX = 620.0


def detect_line(bgr, roi_bottom=0.45, fx=610.0, dist_m=0.8):
    """检测车道两侧黄线, 返回
    (deviation_px, deviation_m, lane_px, ratio) 或 None。
    deviation_px > 0 = 车道中心在画面中心右侧 (车偏左)。
    lane_px = 两线间距 (px)。ratio = 黄像素占比。
    """
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if h < 10 or w < 10:
        return None
    roi = bgr[int(h * (1.0 - roi_bottom)):, :, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    ys, xs = np.where(mask > 0)
    n = len(xs)
    if n < 2 * MIN_LINE_PX:
        return None
    ratio = float(n) / float(mask.size)

    # 全体质心 -> 左右分组 -> 两线质心
    cx_all = float(xs.mean())
    left_sel = xs < cx_all
    right_sel = xs >= cx_all
    if int(left_sel.sum()) < MIN_LINE_PX or int(right_sel.sum()) < MIN_LINE_PX:
        return None          # 只看到一条线 (另一条被遮挡/出画面)
    left_cx = float(xs[left_sel].mean())
    right_cx = float(xs[right_sel].mean())

    lane_px = right_cx - left_cx
    if not (LANE_PX_MIN <= lane_px <= LANE_PX_MAX):
        return None          # 间距不合理: 误检(单条宽线/散点)

    lane_cx = (left_cx + right_cx) / 2.0
    deviation_px = lane_cx - w / 2.0
    if fx <= 0:
        return None
    deviation_m = deviation_px / fx * dist_m
    return deviation_px, deviation_m, lane_px, ratio


def draw_line_debug(bgr, deviation_px, lane_px, ratio):
    """调试图: 画两线质心(蓝)、车道中心(红)、画面中心(绿)。"""
    img = bgr.copy()
    h, w = img.shape[:2]
    lane_cx = int(round(w / 2.0 + deviation_px))
    half = int(round(lane_px / 2.0))
    y0 = int(h * 0.6)
    cv2.line(img, (lane_cx - half, y0), (lane_cx - half, h - 1), (255, 0, 0), 2)
    cv2.line(img, (lane_cx + half, y0), (lane_cx + half, h - 1), (255, 0, 0), 2)
    cv2.line(img, (lane_cx, y0), (lane_cx, h - 1), (0, 0, 255), 2)
    cv2.line(img, (w // 2, y0), (w // 2, h - 1), (0, 255, 0), 1)
    cv2.putText(img, 'dev=%.0fpx lane=%.0fpx r=%.3f' % (deviation_px, lane_px, ratio),
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img
