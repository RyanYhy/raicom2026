#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# orbbec_mine_antitank.py —— RAICOM2026 雷区计数 + 反坦克盘检测 主节点
# ------------------------------------------------------------
# 职责 (方案 §4 子系统 B):
#   1) /mine_detect (String, 载荷 "A"/"B") 触发后, 对雷区做 ~5s 采样扫描,
#      跑 mine.onnx ({0:mine_red 未排雷, 1:mine_blue 已排雷}),
#      跨帧去重 (每区最多 1 盘, 双色同现取高置信) → 上报字节级精确报告串
#      "A雷区未排雷数量x个,已经排雷数量x个" (半角逗号) + 落盘 + 状态话题
#   2) /antitank_check (std_msgs/Empty) 触发后, 跑 anti_tank.onnx ({0:anti_tank}),
#      连续 2 帧 + 单目框宽测距一致确认 → 发 /vision/anti_tank_event +
#      存证据图; 可选项 /vision/obstacle_cloud 点云 (需先标定静态 TF, 默认关)
# 输入/输出:
#   话题 /mine_detect (std_msgs/String) -> /mine_detect_status (std_msgs/String)
#   话题 /antitank_check (std_msgs/Empty) -> /vision/anti_tank_event (std_msgs/String)
#   文件 data/reports/report_mine.txt + data/evidence/AT_*.jpg
# 相机: Orbbec DaBai DCW2, 通过 /camera/color/image_raw 订阅; 无帧时自动拉起
#       dabai_dcw2.launch (必须 DCW2 专用, enable_color 默认 true)。
# 用法:
#   rosrun raicom2026 orbbec_mine_antitank.py              # 等话题触发
#   rosrun raicom2026 orbbec_mine_antitank.py --zone A     # 启动即扫雷区 A 后退出
# ============================================================
import os
import sys
import time
import math
import struct
import signal
import argparse
import threading

# ---- 环境兼容补丁 (必须最前: melodic 的 py2 rospy) ----
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.insert(0, ros_path)

# 注意: 不要在此 sys.path.insert(0, ../src)。那会让源码包 raicom2026
# shadow 掉 devel 里 catkin 生成的同名包(含 srv 生成模块), 导致
#   from raicom2026.srv import GunAim  -> ModuleNotFoundError。
# devel 生成包的 __init__.py 已用 extend_path 把本包 src 并入 __path__,
# 源码模块(yolo_onnx/camera_ros/report/scan_logic)与生成 srv 都能直接解析。

import cv2
import numpy as np
import rospy
from std_msgs.msg import String, Empty, Header
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import sensor_msgs.point_cloud2 as pc2

# cv_bridge 必须 python3 版并提到最前
sys.path.insert(0, '/home/mowen/cv_bridge_ws/devel/lib/python3/dist-packages')
from cv_bridge import CvBridge

import rospkg
from raicom2026.yolo_onnx import YOLOv8ONNX, draw_detections
from raicom2026.report import ReportStore, make_mine_report
from raicom2026.camera_ros import RosCamera, ensure_master
from raicom2026.srv import DetectTask, DetectTaskResponse, GunAim
from raicom2026.mine_avoid import AT_CLASSES, CLASS_RADIUS, project_bbox_mono

# ============================ 路径 ============================
try:
    PKG = rospkg.RosPack().get_path('raicom2026')
except Exception:
    PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINE_MODEL = os.path.join(PKG, 'data', 'models', 'mine.onnx')
AT_MODEL = os.path.join(PKG, 'data', 'models', 'antitank.onnx')
REPORT_DIR = os.path.join(PKG, 'data', 'reports')
EVIDENCE_DIR = os.path.join(PKG, 'data', 'evidence')
PERSON_REPORT_PATH = os.path.join(REPORT_DIR, 'report_person.txt')
MINE_REPORT_PATH = os.path.join(REPORT_DIR, 'report_mine.txt')

# ============================ 相机 ============================
# 2026-08-16: mine 识别改用云台相机(webcam), 不用 Orbbec
MINE_TOPIC = '/usb_cam/image_raw'
WEBCAM_LAUNCH = 'roslaunch raicom2026 usb_cam_mjpeg.launch'
WEBCAM_WAIT = 15
IMG_W_DEF, IMG_H_DEF = 640, 480

# 2026-08-16: antitank 一次性识别仍用 Orbbec 固定朝前 (点云位置已验证正确)
ORBBEC_TOPIC = '/camera/color/image_raw'
AT_CLOUD_TOPIC = '/visual_mine/obstacles'   # costmap visual_mine 观测源订阅

# ============================ 云台预设角度 (2026-08-16) ============================
# mine 识别点云台姿态 (tilt 中心 90=水平, 向下转 = tilt 减小):
#   A 正前方 + 向下10° ;  B 左转90°(pan clamp 175) + 向下20°
MINE_GIMBAL = {
    'A': (70.0, 70.0),
    'B': (175.0, 70.0),
}

# ============================ 模型/置信度 ============================
MINE_RED = 0
MINE_BLUE = 1
AT = 0
CONF_MINE = 0.45         # 地雷最低置信
CONF_AT = 0.45           # 反坦克最低置信
MINE_SCAN_SEC = 5.0      # 每区雷区采样时长 (方案 ≤10s, 取前半段)
AT_SCAN_SEC = 4.0        # 反坦克确认时长
MINE_MIN_FRAMES = 2      # 一块地雷至少出现 2 帧才算 (防单帧误检)
AT_MIN_FRAMES = 2        # 反坦克累计确认帧数
AT_MISS_TOL = 3          # 反坦克确认时允许的连续漏检帧数 (车动/遮挡不断链)
DEDUP_CTR = 0.35         # 跨帧同物中心容差 (幅宽比例; 车微动给裕量)
AT_DISC_M = 0.02         # 反坦克盘直径 (m)
FOCAL_PX = 610.0         # orbbec 640 宽 ~55°HFOV 焦距(px)
AT_DEPTH_OK = 0.20       # 两帧单目测距一致相对容差 (比例, 远距允许更大抖动)

# 触发/状态话题
MINE_TRIGGER_TOPIC = '/mine_detect'
MINE_STATUS_TOPIC = '/mine_detect_status'
AT_TRIGGER_TOPIC = '/antitank_check'
AT_EVENT_TOPIC = '/vision/anti_tank_event'
CLOUD_TOPIC = '/vision/obstacle_cloud'
CAMERA_FRAME = 'camera_color_optical_frame'   # 点云发布帧; 静态 TF 标定后改 odom


class OrbbecMineAntitank:
    def __init__(self, zone_once=None):
        self.zone_once = zone_once
        self.busy = False
        self.busy_lock = threading.Lock()   # 两个订阅回调可能并发, 防 TOCTOU 双扫
        self.publish_cloud = rospy.get_param('~publish_obstacle_cloud', False)

        # ---------- 相机 (云台 webcam) ----------
        try:
            self.cam = RosCamera(MINE_TOPIC, WEBCAM_LAUNCH, wait=WEBCAM_WAIT)
        except Exception as e:
            rospy.logfatal('[mine] 相机初始化失败: %s', e)
            self.cam = None
        self.img_w = self.cam.w if self.cam and self.cam.w else IMG_W_DEF
        self.img_h = self.cam.h if self.cam and self.cam.h else IMG_H_DEF
        self.webcam_ok = self.cam is not None and self.cam.has_frame()
        if not self.webcam_ok:
            rospy.logwarn('[mine] webcam 无帧 -> 雷区只能报空 (TIER-1 降级)')

        # ---------- 云台服务 (mine 识别点转云台) ----------
        # 2026-08-16: 直接绑定 ServiceProxy (不用 wait_for_service, python3 下不可靠)
        self.aim_srv = None
        self.home_srv = None
        self.gimbal_ok = False
        try:
            self.aim_srv = rospy.ServiceProxy('/gun/aim', GunAim)
            self.home_srv = rospy.ServiceProxy('/gun/home', Trigger)
            self.gimbal_ok = True
        except Exception:
            self.gimbal_ok = False
        if not self.gimbal_ok:
            rospy.logwarn('[mine] 云台服务不可用 -> mine 识别不转云台')

        # ---------- 模型 ----------
        try:
            self.mine_model = YOLOv8ONNX(MINE_MODEL, conf_thres=CONF_MINE,
                                         iou_thres=0.45, intra_op_num_threads=2)
            rospy.loginfo('[mine] mine.onnx 加载完成 (识别计数专用)')
        except Exception as e:
            rospy.logfatal('[mine] mine.onnx 加载失败: %s — 检查 data/models/mine.onnx', e)
            sys.exit(1)
        # 2026-08-16: antitank.onnx 恢复加载 —— 一次性服务识别 (不再常驻)
        try:
            self.at_model = YOLOv8ONNX(AT_MODEL, conf_thres=CONF_AT,
                                       iou_thres=0.45, intra_op_num_threads=2)
            rospy.loginfo('[mine] antitank.onnx 加载完成 (一次性服务专用)')
        except Exception as e:
            rospy.logwarn('[mine] antitank.onnx 加载失败: %s (反坦克服务将报空)', e)
            self.at_model = None
        self.focal_px = rospy.get_param('~focal_px', FOCAL_PX)
        # 2026-08-16: 相机外参可配置 (标定用)。cam_x=光心相对 base_link 前向,
        # cam_y=横向(右正); 默认占位值, 标定后用 param 覆写。
        self.cam_x = rospy.get_param('~cam_x', 0.2)
        self.cam_y = rospy.get_param('~cam_y', 0.0)

        # ---------- Orbbec 订阅 + AMCL 定位 (antitank 一次性识别用) ----------
        # 2026-08-16: antitank 仍用 Orbbec 固定朝前识别 (webcam 归 mine)。
        # 直接订阅 /camera/color/image_raw 帧, 服务触发时读最新帧识别。
        self.br = CvBridge()
        self._orbbec_frame = None    # (bgr, stamp) 最新 Orbbec 彩色帧
        rospy.Subscriber(ORBBEC_TOPIC, Image, self._cb_orbbec, queue_size=1)
        # 2026-08-16: 深度图订阅 (antitank 深度测距用; depth_registration=true
        # 配准后 640x480 对齐彩色, frame_id=camera_color_optical_frame)
        self._depth_frame = None     # (depth_m float32米制, stamp)
        self._depth_w = self._depth_h = None
        rospy.Subscriber('/camera/depth/image_raw', Image, self._cb_depth, queue_size=1)
        self._amcl_pose = None       # (x, y, yaw) 车在 map 的位姿
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped,
                         self._cb_amcl_pose, queue_size=1)
        # 2026-08-16: Cartographer 定位模式没有 /amcl_pose (那是 AMCL 发布的),
        # 加订阅 /tracked_pose (Cartographer 发布, tracking_frame=base_link 在 map 系)
        # 作为回退, 否则任务三 _optical_to_map 拿不到位姿 -> no_antitank
        rospy.Subscriber('/tracked_pose', PoseStamped,
                         self._cb_tracked_pose, queue_size=1)
        self.locked_antitank = []    # [(mx, my), ...] 锁定的盘地图坐标
        self.at_cloud_pub = rospy.Publisher(AT_CLOUD_TOPIC, PointCloud2, queue_size=5)
        self._cloud_timer = None     # 识别完后周期性重发点云 (防 persistence 清除)

        # ---------- 报告/状态 ----------
        self.store = ReportStore(PERSON_REPORT_PATH, MINE_REPORT_PATH)
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        self.status_pub = rospy.Publisher(MINE_STATUS_TOPIC, String, queue_size=5)
        self.at_event_pub = rospy.Publisher(AT_EVENT_TOPIC, String, queue_size=5)
        self.cloud_pub = None
        if self.publish_cloud:
            self.cloud_pub = rospy.Publisher(CLOUD_TOPIC, PointCloud2, queue_size=5)
            rospy.logfatal('[mine] ~publish_obstacle_cloud:=true 已开。警告(评审#5): '
                           '点云单点、只发一次、无清除机制; 若静态 TF 缺失/陈旧会把障碍'
                           '放错位置, 误导 local costmap 反而绕错/逼近真盘。仅当 TF '
                           '已标定并验证过才可开启, 否则保持默认关闭。')

        rospy.Subscriber(MINE_TRIGGER_TOPIC, String, self.cb_mine_trigger, queue_size=10)
        rospy.Subscriber(AT_TRIGGER_TOPIC, Empty, self.cb_at_trigger, queue_size=10)
        # 方案D: 服务触发 (nav 主路径) —— 同步一问一答, 报告串直接返回
        self.srv = rospy.Service('/mine_detect_service', DetectTask,
                                 self.srv_mine_cb)
        self.at_srv = rospy.Service('/antitank_check_service', DetectTask,
                                    self.srv_at_cb)
        # 联动视觉避障节点 (2026-08-14 任务三): 检测到反坦克盘后自动开启
        self._avoid_proxy = None

    # ---------------- 地雷证据图 ----------------
    def _save_mine_evidence(self, zone, ev):
        """雷区证据图 (2026-08-14 单图): 检测框最多的一帧, 标出全部盘。"""
        if ev is None or ev[0] is None:
            return
        frm, bx, sc, ids = ev[0], ev[1], ev[2], ev[3]
        name = 'M_%s.jpg' % zone
        img = draw_detections(frm, bx, sc, ids,
                              {MINE_RED: 'mine_red', MINE_BLUE: 'mine_blue'})
        path = os.path.join(EVIDENCE_DIR, name)
        cv2.imwrite(path, img)
        rospy.loginfo('[mine] 证据图 %s', path)

    def run_mine_zone(self, zone):
        zone = (zone or '').strip().upper()
        if zone not in 'AB':
            raise ValueError('非法雷区区名: %r' % zone)
        # 2026-08-16: 云台转到该区预设角度 (A 正前+下10°, B 左转+下10°)
        pan, tilt = MINE_GIMBAL.get(zone, (90.0, 80.0))
        if self.gimbal_ok:
            try:
                self.aim_srv(pan, tilt)
                rospy.sleep(0.6)   # 舵机稳定后再采样
                rospy.loginfo('[mine] %s区 云台转到 pan=%.0f tilt=%.0f', zone, pan, tilt)
            except rospy.ServiceException as e:
                rospy.logwarn('[mine] 云台 aim 失败: %s (用当前画面)', e)
        best_ev = None                       # 检测框最多的一帧 (计数+证据图共用)
        last_seq = -1
        nframes = 0
        t0 = time.time()
        while time.time() - t0 < MINE_SCAN_SEC and not rospy.is_shutdown():
            f, seq = self.cam.get_frame_seq() if self.cam else (None, -1)
            if f is not None and seq != last_seq:
                last_seq = seq               # 只处理新帧, 避免同一物理帧反复计数
                nframes += 1
                boxes, scores, ids = self.mine_model.detect(f)
                if best_ev is None or len(boxes) > best_ev[4]:
                    best_ev = (f, boxes, scores, ids, len(boxes))
            time.sleep(0.03)
        rospy.loginfo('[mine] %s区 采样 %d 帧', zone, nframes)

        # 2026-08-14 用户要求: 不做跨帧去重, 直接用 YOLO 识别结果计数。
        # 取检测框最多的一帧, 红/蓝类别直接累加 (detect 内部已按 CONF_MINE 过滤+NMS)。
        rc = bc = 0
        if best_ev is not None and best_ev[1]:
            for s, c in zip(best_ev[2], best_ev[3]):
                if c == MINE_RED:
                    rc += 1
                elif c == MINE_BLUE:
                    bc += 1

        # 证据图: 单张 (检测框最多的一帧, 标出全部盘)
        self._save_mine_evidence(zone, best_ev)

        report = make_mine_report(zone, rc, bc)
        self.store.add_mine(zone, rc, bc)
        self.status_pub.publish(report)
        rospy.loginfo('[mine] === %s ===', report)
        # 2026-08-14: 排版结果打印到 stdout (start_person.sh 终端 1 实时显示)
        print('')
        print('=' * 52)
        print('             %s 雷区  识别结果' % zone)
        print('-' * 52)
        print('   未排雷(红): %d     已排雷(蓝): %d' % (rc, bc))
        print('=' * 52)
        print('')
        # 2026-08-16: 识别完舵机归位 (home -> pan=90 tilt=90 正前平视)
        if self.gimbal_ok:
            try:
                self.home_srv()
                rospy.loginfo('[mine] %s区 舵机已归位', zone)
            except rospy.ServiceException as e:
                rospy.logwarn('[mine] 舵机归位失败: %s', e)
        return report

    def cb_mine_trigger(self, msg):
        zone = (msg.data or '').strip().upper()
        if zone not in 'AB':
            rospy.logwarn('[mine] 非法雷区区名 %r, 忽略', msg.data)
            return
        with self.busy_lock:
            if self.busy:
                rospy.logwarn('[mine] 忙, 忽略 %s 区触发', zone)
                return
            self.busy = True
        try:
            self.run_mine_zone(zone)
        except Exception as e:
            rospy.logerr('[mine] 雷区扫描异常: %s', e)
        finally:
            self.busy = False

    def srv_mine_cb(self, req):
        """方案D: /mine_detect_service 服务回调 (rospy.Service 独立线程执行)。
        run_mine_zone 自身不检查 busy (话题回调检查), 这里按同一模式加 busy 守卫。"""
        zone = (req.zone or '').strip().upper()
        if zone not in 'AB':
            rospy.logwarn('[mine] 服务收到非法区名 %r', req.zone)
            return DetectTaskResponse(False, '', 'bad_zone:%s' % req.zone)
        with self.busy_lock:
            if self.busy:
                return DetectTaskResponse(False, '', 'busy')
            self.busy = True
        try:
            report = self.run_mine_zone(zone)
        except Exception as e:
            rospy.logerr('[mine] 服务任务异常: %s', e)
            return DetectTaskResponse(False, '', 'error:%s' % e)
        finally:
            self.busy = False
        if report:
            return DetectTaskResponse(True, report, 'ok')
        return DetectTaskResponse(False, '', 'busy_or_abort')

    # ---------------- 反坦克 (2026-08-16: 一次性识别 + 点云发布) ----------------
    # Ⅳ区盘固定, 主程序在 MineB 下一个点(IV_pos)停稳后调一次本服务:
    #   识别两个盘 -> 光轴系单目测距 -> map 坐标锁定 -> 发布点云常驻 map。
    # 不再常驻识别 (visual_mine_avoidance 的反坦克部分已取消)。
    def _cb_orbbec(self, msg):
        """缓存最新 Orbbec 彩色帧 (antitank 识别用)。"""
        try:
            self._orbbec_frame = (self.br.imgmsg_to_cv2(msg, 'bgr8'),
                                  (msg.header.stamp.secs, msg.header.stamp.nsecs))
        except Exception:
            pass

    def _cb_depth(self, msg):
        """缓存最新深度图 (米制 float32, antitank 深度测距用)。"""
        try:
            enc = msg.encoding or ''
            if '16UC1' in enc or 'mono16' in enc or 'Y11' in enc:
                dm = self.br.imgmsg_to_cv2(msg, '16UC1').astype(np.float32) / 1000.0
            elif '32FC1' in enc:
                dm = self.br.imgmsg_to_cv2(msg, '32FC1')
            else:
                return
            self._depth_frame = (dm, (msg.header.stamp.secs, msg.header.stamp.nsecs))
            self._depth_h, self._depth_w = dm.shape[:2]
        except Exception:
            pass

    def _cb_amcl_pose(self, msg):
        """缓存 AMCL 定位 (map 系车位姿), 用于光轴系->map 系变换。"""
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._amcl_pose = (msg.pose.pose.position.x,
                           msg.pose.pose.position.y, yaw)

    def _cb_tracked_pose(self, msg):
        """缓存 Cartographer 定位位姿 (tracking_frame=base_link 在 map 系)。
        2026-08-16: Cartographer 模式下 /amcl_pose 无数据, 用 /tracked_pose 回退。
        base_link 与 base_footprint XY 一致 (仅 z 差 0.082m), 直接当车位姿用。"""
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._amcl_pose = (msg.pose.position.x, msg.pose.position.y, yaw)

    def _optical_to_map(self, X, Z):
        """光轴系盘位置 (X右, Z前=距离) -> map 系 (x,y)。
        相机外参 cam_x/cam_y 可配置 (标定后覆写)。返回 (mx, my) 或 None(无定位)。"""
        if self._amcl_pose is None:
            return None
        x_c, y_c, yaw = self._amcl_pose
        bx = self.cam_x + Z
        by = self.cam_y - X
        mx = x_c + math.cos(yaw) * bx - math.sin(yaw) * by
        my = y_c + math.sin(yaw) * bx + math.cos(yaw) * by
        return (mx, my)

    def _median_depth(self, depth_m, dx, dy, pad=2):
        """取深度图 (dx,dy) 周围 (2*pad+1)^2 的有效值中位数 (抗反光孔洞)。
        返回 float 或 None(全无效)。"""
        h, w = depth_m.shape[:2]
        x0, x1 = max(0, dx - pad), min(w, dx + pad + 1)
        y0, y1 = max(0, dy - pad), min(h, dy + pad + 1)
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[(patch > 0.05) & np.isfinite(patch)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _bbox_to_optical_depth(self, b, depth_m):
        """检测框 -> 光轴系 (X, Z) 用深度图 (底边中心 5x5 中位数深度)。
        平视相机: 框底边=盘贴地点, 取底边深度投影到地面平面。
        深度无效返回 (None, None)。"""
        left, top, w, h = b
        cx = int(np.clip(left + w / 2.0, 0, 639))
        cy = int(np.clip(top + h, 0, 479))     # 底边 (盘贴地)
        # 深度图可能和彩色分辨率不同, 按比例缩放 (配准后通常 640x480 一致)
        if self._depth_w and self._depth_w != 640:
            dx = int(cx * self._depth_w / 640.0)
            dy = int(cy * self._depth_h / 480.0)
        else:
            dx, dy = cx, cy
        d = self._median_depth(depth_m, dx, dy)
        if d is None or d < 0.05 or d > 3.0:
            return (None, None)
        X = (cx - 320.0) / self.focal_px * d
        return (float(X), float(d))

    def _run_antitank_once(self):
        """扫描 Orbbec 帧 ~AT_SCAN_SEC, 识别两个 antitank 盘, 锁定 map 坐标。
        返回锁定的盘数量。"""
        if self.at_model is None:
            rospy.logwarn('[mine] antitank.onnx 未加载, 反坦克识别跳过')
            return 0
        t0 = time.time()
        last_stamp = None
        nframes = 0
        best_ev = None   # 检测框最多的帧 (证据图用)
        while time.time() - t0 < AT_SCAN_SEC and not rospy.is_shutdown():
            if self._orbbec_frame is None:
                rospy.logwarn_throttle(3.0, '[mine] 等待 Orbbec 帧 ...')
                time.sleep(0.05)
                continue
            bgr, stamp = self._orbbec_frame
            if stamp == last_stamp:
                time.sleep(0.03)
                continue
            last_stamp = stamp
            nframes += 1
            boxes, scores, ids = self.at_model.detect(bgr)
            rospy.loginfo_throttle(2.0, '[mine] at推理: 检测%d个 (帧%d)',
                                   len(boxes), nframes)
            if best_ev is None or len(boxes) > best_ev[4]:
                best_ev = (bgr, boxes, scores, ids, len(boxes))
            for b, s, c in zip(boxes, scores, ids):
                key = AT_CLASSES.get(int(c))
                if key is None or s < CONF_AT:
                    continue
                # 2026-08-16: 深度优先测距 (框底边中心 5x5 中位数深度, 配准后对齐彩色);
                # 深度无效(反光孔洞)回退单目框宽测距。
                X = Z = None
                if self._depth_frame is not None:
                    X, Z = self._bbox_to_optical_depth(b, self._depth_frame[0])
                if X is None:
                    pt = project_bbox_mono(b, self.focal_px, self.focal_px,
                                           320.0, 240.0, AT_DISC_M, 3.0)
                    if pt is None:
                        continue
                    X, Z = pt[0], pt[2]
                mp = self._optical_to_map(X, Z)
                if mp is None:
                    continue
                # 去重阈值 0.14m (两盘分开锁定, 防单目测距误差合并)
                if not any((mp[0] - lx) ** 2 + (mp[1] - ly) ** 2 < 0.02
                           for lx, ly in self.locked_antitank):
                    self.locked_antitank.append(mp)
                    rospy.loginfo('[mine] 锁定反坦克盘 map 坐标: (%.2f, %.2f) conf=%.2f 距离=%.2fm',
                                  mp[0], mp[1], s, Z)
            time.sleep(0.03)
        # 2026-08-16: 保存反坦克证据图 (检测框最多的一帧)
        self._save_at_evidence(best_ev)
        self._publish_at_cloud()
        # 2026-08-16: 识别完后周期性重发点云 (0.5s 一次), 防止 costmap
        # observation_persistence 到期清除障碍标记 (一次性服务不再常驻重发)
        if self._cloud_timer is None:
            self._cloud_timer = rospy.Timer(rospy.Duration(0.5), self._cb_cloud_timer)
            rospy.loginfo('[mine] 已启动点云周期重发 (0.5s)')
        rospy.loginfo('[mine] 反坦克一次性识别完成: %d 个盘 (%d 帧)',
                      len(self.locked_antitank), nframes)
        return len(self.locked_antitank)

    def _cb_cloud_timer(self, event):
        """周期性重发锁定盘点云 (2026-08-16)。"""
        if self.locked_antitank:
            self._publish_at_cloud()

    def _save_at_evidence(self, ev):
        """反坦克证据图 (2026-08-16): 检测框最多的一帧, 标出全部盘。"""
        if ev is None or ev[0] is None:
            return
        frm, bx, sc, ids = ev[0], ev[1], ev[2], ev[3]
        img = draw_detections(frm, bx, sc, ids, AT_CLASSES)
        path = os.path.join(EVIDENCE_DIR, 'AT_iv.jpg')
        cv2.imwrite(path, img)
        rospy.loginfo('[mine] 反坦克证据图 %s', path)

    def _publish_at_cloud(self):
        """把锁定的盘 map 坐标发布为点云 (map 帧, 球面点集, 不跟车)。"""
        header = Header()
        header.frame_id = 'map'
        header.stamp = rospy.Time.now()
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        points = []
        r = 0.015   # 2026-08-16: 点云半径 5cm (用户指定, 以映射点为圆心的圆形点云)
        for (mx, my) in self.locked_antitank:
            for i in range(20):
                angle = 2.0 * math.pi * i / 20.0
                points.append((mx + r * math.cos(angle),
                               my + r * math.sin(angle), 0.01))
        try:
            msg = pc2.create_cloud(header, fields, points)
            self.at_cloud_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, '[mine] cloud 发布失败: %s', e)

    def srv_at_cb(self, req):
        """/antitank_check_service 服务回调 (2026-08-16 重构):
        一次性识别两个 antitank 盘 -> 计算 map 坐标 -> 发布两个点云。
        (不再常驻识别; 盘固定, 一次识别后点云常驻 map 供 costmap 避障)"""
        with self.busy_lock:
            if self.busy:
                return DetectTaskResponse(False, '', 'busy')
            self.busy = True
        try:
            n = self._run_antitank_once()
        except Exception as e:
            rospy.logerr('[mine] 反坦克服务异常: %s', e)
            return DetectTaskResponse(False, '', 'error:%s' % e)
        finally:
            self.busy = False
        if n > 0:
            return DetectTaskResponse(True, 'antitank=%d' % n, 'ok')
        return DetectTaskResponse(False, '', 'no_antitank')

    def cb_at_trigger(self, msg):
        with self.busy_lock:
            if self.busy:
                rospy.logwarn('[mine] 忙, 忽略反坦克触发')
                return
            self.busy = True
        try:
            self._run_antitank_once()
        except Exception as e:
            rospy.logerr('[mine] 反坦克触发异常: %s', e)
        finally:
            self.busy = False

    def shutdown(self):
        if self.cam is not None:
            self.cam.shutdown()


def _sig_to_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description='雷区+反坦克检测节点')
    parser.add_argument('--zone', default=None, help='启动即扫雷区并退出 (A/B)')
    args = parser.parse_args(rospy.myargv(sys.argv)[1:])   # 剥掉 __name:=/__log:= 等 ROS 参数

    ensure_master()   # 必须在 init_node 之前: 无 master 时 init 会直接崩
    rospy.init_node('orbbec_mine_antitank', anonymous=False)
    signal.signal(signal.SIGTERM, _sig_to_interrupt)
    signal.signal(signal.SIGINT, _sig_to_interrupt)

    node = OrbbecMineAntitank(zone_once=args.zone)
    try:
        if args.zone:
            node.run_mine_zone(args.zone)
        else:
            rospy.loginfo('[mine] 就绪, 等待 /mine_detect (A/B) 或 /antitank_check')
            rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rospy.loginfo('[mine] 退出')


if __name__ == '__main__':
    main()
