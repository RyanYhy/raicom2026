#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# visual_mine_avoidance.py —— RAICOM2026 视觉雷区/反坦克盘避障 常驻节点
# ------------------------------------------------------------
# 职责: 导航期间持续检测雷区盘 (mine.onnx: 红=未排雷/蓝=已排雷) 与
#       反坦克盘 (antitank.onnx), 把确认后的盘位置展开为球面点集, 发布为
#       /visual_mine/obstacles (PointCloud2, camera_color_optical_frame 帧),
#       由 move_base costmap 的 visual_mine 观察源注入 obstacle_layer,
#       规划器自动绕行 (local+global 双注入)。
#
# 关键设计 (方案避障子系统):
#   - 点云发在传感器帧 camera_color_optical_frame, 节点零动态 TF:
#     costmap_2d 的 C++ tf2 完成 sensor->odom/map 变换 (与 laser scan 同构)。
#     py3 下 tf2_ros/tf.transformations 均不可用, 绝不 import。
#   - 球面点集 (Fibonacci) 在任一姿态下 XY 投影≈半径 R 圆盘, 免疫车俯仰。
#   - 每类独立 MineTracker 时序确认 (min_hits=2, max_miss=8)。
#   - 模型缺失 FAIL-FAST (logfatal+exit(1)); 相机/参数缺失降级不退出。
#   - 启动后每 3s 自检 /move_base/local_costmap/observation_sources 是否
#     含 visual_mine; 未注入 loud logwarn 提示必须用 nav_competition_2026.launch。
#
# 用法:
#   roslaunch raicom2026 visual_mine_avoidance.launch      # bench/独立避障
#   roslaunch raicom2026 nav_competition_2026.launch       # 比赛主入口
# ============================================================
import os
import sys
import time
import math
import signal
import argparse

# ---- 环境兼容补丁 (必须最前: melodic 的 py2 rospy) ----
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.insert(0, ros_path)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import cv2
import rospy
from std_msgs.msg import Header, String, Float32, Bool
from std_srvs.srv import SetBool, SetBoolResponse
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2

# cv_bridge 必须 python3 版并提到最前
sys.path.insert(0, '/home/mowen/cv_bridge_ws/devel/lib/python3/dist-packages')
from cv_bridge import CvBridge

import rospkg
from raicom2026.yolo_onnx import YOLOv8ONNX, draw_detections
from raicom2026.camera_ros import ensure_master
from raicom2026.mine_avoid import (
    MINE_CLASSES, AT_CLASSES, CLASS_RADIUS, CLASS_COLOR,
    depth_mm_to_m, project_bbox_to_optical, project_bbox_mono,
    make_ring_points_optical,
    MineTracker, check_observation_source,
)
from raicom2026.line_detect import detect_line, draw_line_debug

# 禁 import: torch / tf2_ros / tf / image_geometry (py3 全不可用/不允许)


# ============================ 路径 ============================
try:
    PKG = rospkg.RosPack().get_path('raicom2026')
except Exception:
    PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINE_MODEL = os.path.join(PKG, 'data', 'models', 'mine.onnx')
AT_MODEL = os.path.join(PKG, 'data', 'models', 'antitank.onnx')

# 订阅/发布话题
RGB_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/depth/image_raw'
INFO_TOPIC = '/camera/color/camera_info'
CLOUD_TOPIC = '/visual_mine/obstacles'
MARKERS_TOPIC = '/visual_mine/markers'
STATUS_TOPIC = '/visual_mine/status'
ZONE_TOPIC = '/nav_competition/zone'   # nav_competition 发布的当前站名 (反坦克盘门控)

# 默认内参回退 (orbbec 640 宽 ~55°HFOV, 与 orbbec_mine_antitank 一致)
FOCAL_PX = 610.0
PRINCIPAL_X, PRINCIPAL_Y = 320.0, 240.0

# 反坦克盘直径 (m) —— 检测框宽反推距离用 (2026-08-15 一致性校验)
AT_DISC_M = 0.10


class VisualMineAvoidance:
    def __init__(self):
        # ---------- 参数 (priv ns) ----------
        self.mine_model_path = rospy.get_param('~mine_model_path', MINE_MODEL)
        self.at_model_path = rospy.get_param('~at_model_path', AT_MODEL)
        self.mine_conf = rospy.get_param('~mine_conf', 0.5)
        self.at_conf = rospy.get_param('~at_conf', 0.5)
        self.iou = rospy.get_param('~iou', 0.45)
        self.intra_threads = rospy.get_param('~intra_op_num_threads', 2)
        self.mine_radius = rospy.get_param('~mine_radius', 0.12)
        self.at_radius = rospy.get_param('~at_radius', 0.15)
        self.cloud_n_pts = rospy.get_param('~cloud_n_pts', 20)
        self.max_range = rospy.get_param('~max_range', 1.0)
        self.publish_rate = rospy.get_param('~publish_rate', 10.0)
        self.min_hits = rospy.get_param('~tracker_min_hits', 2)
        self.max_miss = rospy.get_param('~tracker_max_miss', 8)
        self.smooth = rospy.get_param('~track_smooth', 0.5)
        self.stale = rospy.get_param('~track_stale', 2.0)
        self.rgb_topic = rospy.get_param('~rgb_topic', RGB_TOPIC)
        self.depth_topic = rospy.get_param('~depth_topic', DEPTH_TOPIC)
        self.info_topic = rospy.get_param('~info_topic', INFO_TOPIC)
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.depth_scale_x = rospy.get_param('~depth_scale_x', 1.0)
        self.depth_scale_y = rospy.get_param('~depth_scale_y', 400.0 / 480.0)
        self.depth_min = rospy.get_param('~depth_min', 0.05)
        self.depth_patch = rospy.get_param('~depth_patch', 3)
        self.at_infer_interval = rospy.get_param('~at_infer_interval', 3)
        # 反坦克盘推理门控: 仅当 nav 当前站名在 at_gate_zones 内时启用 anti_tank 推理。
        # 空列表 = 始终启用 (bench 独立运行/无 nav 场景)。配置示例: ["IV_pos"]
        self.at_gate_zones = list(rospy.get_param('~at_gate_zones', []))
        self.auto_launch_camera = rospy.get_param('~auto_launch_camera', False)
        self.use_camera_info = rospy.get_param('~use_camera_info', True)
        self.fx = rospy.get_param('~fx', FOCAL_PX)
        self.fy = rospy.get_param('~fy', FOCAL_PX)
        self.cx = rospy.get_param('~cx', PRINCIPAL_X)
        self.cy = rospy.get_param('~cy', PRINCIPAL_Y)
        self.obs_source_key = rospy.get_param('~obs_source_key',
                                              '/move_base/local_costmap/observation_sources')

        # 覆盖: 半径优先取用户显式参数 (与 CLASS_RADIUS 默认一致)
        CLASS_RADIUS['mine_red'] = CLASS_RADIUS['mine_blue'] = self.mine_radius
        CLASS_RADIUS['anti_tank'] = self.at_radius

        # ---------- 模型 (FAIL-FAST) ----------
        # 2026-08-16: 反坦克常驻识别禁用 —— 改由 orbbec_mine_antitank 的
        # /antitank_check_service 一次性识别 (MineB 下一个点停稳后调用)。
        # 本节点保留为"黄线循线"节点。~enable_at 仅调试用 (默认关)。
        self.enable_at = rospy.get_param('~enable_at', False)
        if self.enable_at:
            try:
                self.at_model = YOLOv8ONNX(self.at_model_path, conf_thres=self.at_conf,
                                           iou_thres=self.iou, intra_op_num_threads=self.intra_threads)
                rospy.loginfo('[avoid] antitank.onnx 加载完成 (常驻避障, 调试用)')
            except Exception as e:
                rospy.logfatal('[avoid] 模型加载失败: %s — 检查 data/models/antitank.onnx', e)
                sys.exit(1)
        else:
            self.at_model = None
            rospy.loginfo('[avoid] 反坦克常驻识别已禁用 (改用一次性服务); 本节点只做循线')

        # ---------- 订阅/发布 ----------
        self.br = CvBridge()
        self._rgb = None
        self._depth = None
        self._info = None
        self._last_stamp = (0, 0)   # (secs, nsecs) 帧去重键
        self._t0 = time.time()
        self._tick = 0
        self._marker_id = 0
        self._last_status_t = 0.0
        self.trackers = {}
        self._last_confirmed = []       # 非推理 tick 复用上次 confirmed (防闪)
        self._zone = None            # nav 当前站名 (None=还没收到)
        self._at_gate_state = None   # 门控状态缓存 (None/True/False), 用于切换日志
        # 2026-08-16: 盘位置锁定 —— 第一次识别后固定在地图坐标, 不跟车
        self._amcl_pose = None          # (x, y, yaw) 车在 map 的位姿
        self.locked_antitank = []       # [(mx, my), ...] 锁定的盘地图坐标
        # 2026-08-14: 开关服务 (任务三) —— 关闭时不推理、发空云清 costmap
        self.enabled = rospy.get_param('~start_enabled', True)
        self.enable_srv = rospy.Service('/visual_mine/set_enabled', SetBool,
                                        self._cb_set_enabled)

        rospy.Subscriber(self.rgb_topic, Image, self._cb_rgb, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self._cb_depth, queue_size=1)
        rospy.Subscriber(self.info_topic, CameraInfo, self._cb_info, queue_size=5)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped,
                         self._cb_amcl_pose, queue_size=1)
        if self.at_gate_zones:
            rospy.Subscriber(ZONE_TOPIC, String, self._cb_zone, queue_size=5)

        self.cloud_pub = rospy.Publisher(CLOUD_TOPIC, PointCloud2, queue_size=5)
        self.marker_pub = rospy.Publisher(MARKERS_TOPIC, MarkerArray, queue_size=5)
        self.status_pub = rospy.Publisher(STATUS_TOPIC, String, queue_size=5)
        # 2026-08-14 循线: 黄线横向偏差 (米, 0=线居中) + 有效性 + 调试图
        self.line_dev_pub = rospy.Publisher('/visual_line/deviation', Float32,
                                            queue_size=5)
        self.line_valid_pub = rospy.Publisher('/visual_line/valid', Bool,
                                              queue_size=5)
        self.line_dbg_pub = rospy.Publisher('/visual_line/debug_image', Image,
                                            queue_size=2)
        # 2026-08-15: antitank 检测框实时显示 (image_view 看模型框了什么)
        self.at_debug_pub = rospy.Publisher('/visual_mine/at_debug', Image,
                                            queue_size=2)

        # ---------- 相机自动拉起 (默认关; 比赛由 nav launch 显式起 driver) ----------
        if self.auto_launch_camera:
            try:
                from raicom2026.camera_ros import RosCamera
                ORBBEC_LAUNCH = ('roslaunch orbbec_camera dabai_dcw2.launch '
                                 'enable_depth:=true enable_ir:=false '
                                 'enable_point_cloud:=false')
                self._cam_guard = RosCamera(self.rgb_topic, ORBBEC_LAUNCH, wait=20)
            except Exception as e:
                rospy.logwarn('[avoid] 自动拉起 orbbec 失败(可能已在跑): %s', e)
                self._cam_guard = None
        else:
            self._cam_guard = None

        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self._process)

        # 就绪
        self._check_costmap()
        rospy.loginfo('[avoid] 就绪. cloud=%s(%s), mine_conf=%.2f at_conf=%.2f, '
                      'max_range=%.2f, publish_rate=%.1fHz',
                      CLOUD_TOPIC, self.camera_frame, self.mine_conf, self.at_conf,
                      self.max_range, self.publish_rate)

    # ---------------- 订阅回调 ----------------
    def _cb_set_enabled(self, req):
        """视觉避障开关 (2026-08-14 任务三): true 开启实时检测+点云;
        false 关闭 (清 trackers + 停推理 + 发空云)。全程走完后调用关闭。"""
        self.enabled = bool(req.data)
        if not self.enabled:
            self.trackers = {}
            self.locked_antitank = []    # 关闭时清锁定的盘
            self._last_stamp = (0, 0)
        rospy.loginfo('[avoid] 视觉避障: %s', '开启' if self.enabled else '关闭')
        print('=' * 52)
        print('    视觉避障: %s' % ('已开启 (实时检测+点云)' if self.enabled
                                     else '已关闭 (停止发布障碍点云)'))
        print('=' * 52)
        return SetBoolResponse(success=True, message='enabled=%s' % self.enabled)

    def _cb_rgb(self, msg):
        self._rgb = msg

    def _cb_depth(self, msg):
        self._depth = msg

    def _cb_info(self, msg):
        self._info = msg

    def _cb_zone(self, msg):
        """nav 站名更新 (四区门控输入)。"""
        if msg.data != self._zone:
            self._zone = msg.data
            self._at_gate_state = None   # 强制重新记录门控切换日志

    def _cb_amcl_pose(self, msg):
        """缓存 AMCL 定位 (map 系车位姿), 用于光轴系->map 系手动变换。"""
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._amcl_pose = (msg.pose.pose.position.x,
                           msg.pose.pose.position.y, yaw)

    def _optical_to_map(self, X, Z):
        """光轴系盘位置 (X右, Z前=距离) -> map 系 (x,y)。
        相机相对 base_link: 前 0.2m, 高 0.2m, 平视(光轴朝前)。
        光轴 Z(前) -> base_link x(前); 光轴 X(右) -> base_link -y(左)。
        再按车 yaw 旋转到 map 系。返回 (mx, my) 或 None(无定位)。"""
        if self._amcl_pose is None:
            return None
        x_c, y_c, yaw = self._amcl_pose
        bx = 0.2 + Z                       # base_link x (相机前0.2 + 盘距离)
        by = -X                            # base_link y (盘横向, 右为负)
        mx = x_c + math.cos(yaw) * bx - math.sin(yaw) * by
        my = y_c + math.sin(yaw) * bx + math.cos(yaw) * by
        return (mx, my)

    def _at_enabled(self):
        """反坦克盘推理门控: 仅当当前站名在 at_gate_zones 内启用。
        at_gate_zones 为空 = 始终启用 (bench/单独测避障)。
        2026-08-15 修正: 尚未收到 nav 站名 (_zone is None, prep/赛前)
        = 不推理 (前面三个区域没有 antitank, 省资源)。"""
        if not self.at_gate_zones:
            return True
        if self._zone is None:
            return False
        on = self._zone in self.at_gate_zones
        if on != self._at_gate_state:
            self._at_gate_state = on
            if on:
                rospy.loginfo('[avoid] 进入反坦克盘检测区 (zone=%s), 启用 anti_tank 推理',
                              self._zone)
            else:
                rospy.loginfo('[avoid] 离开反坦克盘检测区 (zone=%s), 暂停 anti_tank 推理',
                              self._zone)
        return on

    # ---------------- 主循环 ----------------
    def _depth_to_m(self, msg):
        """Image -> float32 米制深度图; 支持 16UC1(毫米) / 32FC1(米)。"""
        enc = (msg.encoding or '')
        try:
            if '16UC1' in enc or 'mono16' in enc or 'Y11' in enc:
                return depth_mm_to_m(self.br.imgmsg_to_cv2(msg, '16UC1'))
            if '32FC1' in enc or '32F' in enc:
                img = self.br.imgmsg_to_cv2(msg, '32FC1')
                return img.astype(np.float32)
            rospy.logwarn_throttle(5.0, '[avoid] 未知 depth encoding: %s', enc)
            return None
        except Exception as e:
            rospy.logwarn_throttle(3.0, '[avoid] depth 转换失败: %s', e)
            return None

    def _intrinsics(self):
        info = self._info
        if self.use_camera_info and info is not None:
            fx, fy = info.K[0], info.K[4]
            cx0, cy0 = info.K[2], info.K[5]
            if fx > 0 and fy > 0:
                return fx, fy, cx0, cy0
        return self.fx, self.fy, self.cx, self.cy

    def _process(self, event):
        self._tick += 1
        if not self.enabled:
            # 关闭态: 不发障碍点云 (costmap persistence 到期后障碍自然消失)
            return
        # 2026-08-15: 单目测距后不再依赖 depth 图, 只等 rgb 帧即可
        if self._rgb is None:
            if self._tick % 20 == 0:   # ~2s
                rospy.logwarn_throttle(3.0, '[avoid] 等待 rgb 帧 ...')
            return
        # 同一物理帧不重复处理: 用 header.stamp 作键 (Image 顶层无 seq 字段,
        # header.seq 驱动未必自增, stamp 恒唯一)
        st = (self._rgb.header.stamp.secs, self._rgb.header.stamp.nsecs)
        if st == self._last_stamp:
            return
        self._last_stamp = st

        try:
            bgr = self.br.imgmsg_to_cv2(self._rgb, 'bgr8')
        except Exception:
            return
        fx, fy, cx0, cy0 = self._intrinsics()

        # 0) 黄线循线检测 (2026-08-14 v2): 车道两侧双黄线, 车走中间。
        #    每帧发布车道中心横向偏差, 供 cmd_vel_merge 叠加横向修正
        #    (无线/单线/间距异常时 valid=False, 修正自动失效)。
        try:
            line = detect_line(bgr, fx=fx)
            if line is not None:
                _dev_px, dev_m, _lane_px, ratio = line
                self.line_dev_pub.publish(Float32(dev_m))
                self.line_valid_pub.publish(Bool(True))
                if self._tick % 2 == 0:
                    self.line_dbg_pub.publish(
                        self.br.cv2_to_imgmsg(
                            draw_line_debug(bgr, _dev_px, _lane_px, ratio),
                            'bgr8'))
            else:
                self.line_dev_pub.publish(Float32(0.0))
                self.line_valid_pub.publish(Bool(False))
        except Exception as e:
            rospy.logwarn_throttle(5.0, '[avoid] 黄线检测异常: %s', e)

        # 2026-08-16: 反坦克常驻识别禁用 —— 只做循线, 跳过 antitank 推理/点云发布
        # (antitank 改由 orbbec_mine_antitank 的一次性服务在 IV_pos 停稳后识别)
        if not self.enable_at:
            return

        # 1) antitank.onnx 每 at_infer_interval tick (省 CPU; 2026-08-14 只做反坦克避障)
        raw = []                         # [(class_key, (x,y,z), conf)]
        infer_tick = (self._at_enabled() and
                      self._tick % self.at_infer_interval == 0)
        if infer_tick:
            boxes, scores, ids = self.at_model.detect(bgr)
            # 2026-08-15: 打印每个检测框坐标, 用于判断"假检测(框固定跟车)"
            # vs "真检测(框随盘动)" —— 框(x,y,w,h) 固定 = 模型误检固定元素
            box_desc = ' '.join('(%d,%d,%dx%d)' % (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                                for b in boxes)
            rospy.loginfo_throttle(3.0, '[avoid] at推理: 检测%d个 conf>=%.2f (tick=%d, zone=%s) 框: %s',
                                   len(boxes), self.at_conf, self._tick, self._zone, box_desc)
            # 2026-08-15: 发布检测框调试图 (image_view 实时看模型框了什么)
            try:
                dbg = draw_detections(bgr, boxes, scores, ids, AT_CLASSES)
                self.at_debug_pub.publish(self.br.cv2_to_imgmsg(dbg, 'bgr8'))
            except Exception as e:
                rospy.logwarn_throttle(5.0, '[avoid] at_debug 发布失败: %s', e)
            for b, s, c in zip(boxes, scores, ids):
                key = AT_CLASSES.get(int(c))
                if key is None or s < self.at_conf:
                    continue
                # 2026-08-15 纯单目测距映射: 框在画面中的位置 + 框宽反推距离
                # -> pinhole 投影到相机光轴系 3D 点。不用深度图(Orbbec 深度
                # depth_registration=False 与彩色不对齐+孔洞+反光, 会固定错位置)。
                pt = project_bbox_mono(b, fx, fy, cx0, cy0, AT_DISC_M,
                                       self.max_range)
                if pt is not None:
                    raw.append((key, pt, float(s)))

        # 2) 每类独立 MineTracker 时序确认
        #    2026-08-15 关键修复: 之前的循环只遍历"本帧检测到的类别"(by_class),
        #    无检测时 by_class 为空 -> tracker.update 不被调用 -> miss 永不累计
        #    -> confirmed 的 track 永不丢失 -> 盘拿走/检测不到后点云永远停在
        #    最后一次位置(相机系, 跟车走) -> "点云固定在小车前方"。
        #    修复: 所有已知 tracker 都要 update (无检测的类别传空列表, miss 才累计,
        #    连续 max_miss 次没检测到就 lost, 点云才消失)。
        by_class = {}
        for key, pt, conf in raw:
            by_class.setdefault(key, []).append((pt[0], pt[1], pt[2], conf))
        if infer_tick:
            all_keys = set(self.trackers.keys()) | set(by_class.keys())
            for key in all_keys:
                tr = self.trackers.get(key)
                if tr is None:
                    tr = MineTracker(min_hits=self.min_hits, max_miss=self.max_miss,
                                     match_radius=3.0 * CLASS_RADIUS[key],
                                     smooth=self.smooth, stale=self.stale,
                                     clock=time.time)
                    self.trackers[key] = tr
                tr.update(by_class.get(key, []))   # 无检测 = 空列表, miss 累计
            # 2026-08-16: confirmed 的光轴系坐标 -> map 系, 第一次后锁定
            # (点云固定在地图坐标, 不跟车, 之后不再更新位置)
            for key, tr in self.trackers.items():
                for x, _y, z, _conf in tr.confirmed_positions():
                    mp = self._optical_to_map(x, z)
                    if mp is None:
                        continue
                    if not any((mp[0] - lx) ** 2 + (mp[1] - ly) ** 2 < 0.09
                               for lx, ly in self.locked_antitank):
                        self.locked_antitank.append(mp)
                        rospy.loginfo('[avoid] 锁定反坦克盘 map 坐标: (%.2f, %.2f)',
                                      mp[0], mp[1])

        # 3) 发布 (锁定坐标固定在地图, 不跟车)
        self._publish_cloud()
        self._publish_markers()
        self._publish_status()
        self._check_costmap()

    # ---------------- 发布 ----------------
    def _publish_cloud(self):
        header = Header()
        header.frame_id = 'map'          # 2026-08-16: 锁定的地图坐标
        header.stamp = rospy.Time.now()
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        points = []
        r = CLASS_RADIUS['anti_tank']
        for (mx, my) in self.locked_antitank:
            for i in range(self.cloud_n_pts):
                angle = 2.0 * math.pi * i / float(self.cloud_n_pts)
                points.append((mx + r * math.cos(angle),
                               my + r * math.sin(angle),
                               0.01))
        try:
            msg = pc2.create_cloud(header, fields, points)
            self.cloud_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, '[avoid] cloud 发布失败: %s', e)

    def _publish_markers(self):
        arr = MarkerArray()
        r = CLASS_RADIUS['anti_tank']
        for (mx, my) in self.locked_antitank:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = rospy.Time.now()
            m.ns = 'mine_obstacle'
            m.id = self._marker_id
            self._marker_id += 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.scale.x = m.scale.y = 2.0 * r
            m.scale.z = 0.03
            cr, cg, cb = CLASS_COLOR['anti_tank']
            m.color.r, m.color.g, m.color.b, m.color.a = cr, cg, cb, 0.7
            m.lifetime = rospy.Duration(0.5)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = mx, my, 0.0
            m.pose.orientation.w = 1.0
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def _publish_status(self):
        if time.time() - self._last_status_t < 1.0:
            return
        self._last_status_t = time.time()
        total = sum(len(t.tracks) for t in self.trackers.values())
        self.status_pub.publish(String(
            'locked=%d total=%d class=anti_tank' % (len(self.locked_antitank), total)))

    def _check_costmap(self):
        """每 ~3s 自检 observation_sources 是否含 visual_mine。"""
        if self._tick % max(1, int(self.publish_rate * 3)) != 0:
            return
        val = rospy.get_param(self.obs_source_key, '')
        if not check_observation_source(val):
            rospy.logwarn_throttle(
                5.0, '[avoid] %s = %r 未注入 visual_mine! 避障点云不会被 costmap 使用。'
                     '必须用 raicom2026/nav_competition_2026.launch 起 move_base, '
                     '不要用 nav_multi.launch', self.obs_source_key, val)

    def shutdown(self):
        if getattr(self, '_cam_guard', None) is not None:
            try:
                self._cam_guard.shutdown()
            except Exception:
                pass


def _sig_to_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description='视觉雷区/反坦克避障节点')
    args, _ = parser.parse_known_args()

    ensure_master()   # 必须在 init_node 之前: 无 master 时 init 会直接崩
    rospy.init_node('visual_mine_avoidance', anonymous=False)
    signal.signal(signal.SIGTERM, _sig_to_interrupt)
    signal.signal(signal.SIGINT, _sig_to_interrupt)

    node = VisualMineAvoidance()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rospy.loginfo('[avoid] 退出')


if __name__ == '__main__':
    main()
