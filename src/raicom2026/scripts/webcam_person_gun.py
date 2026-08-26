#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# webcam_person_gun.py —— RAICOM2026 人物识别 + 云台打靶 主节点
# ------------------------------------------------------------
# 职责 (方案 §3 子系统 A):
#   1) 触发后对指定区 A~D 做"云台两档俯仰 × 5 位 pan"扫描, 跑 person.onnx
#      ({0:hostage, 1:enemy, 2:friend}), 把每个检测映射成绝对云台角
#   2) 聚类去重 (scan_logic.cluster_person_detections) → 各类型人数
#   3) 立即上报字节级精确报告串 (report.py), 落盘 report_person.txt + 发状态话题
#   4) 只对 enemy 逐个瞄准/安全门/闭环微调/点射/击倒验证 (0.05s burst, 3s 冷却)
#   5) 每个目标存 1 张标注证据图 (框+类别+置信度+绝对角)
# 输入:
#   话题 /person_detect (std_msgs/String, 载荷 "A"~"D") —— nav 触发
# 输出:
#   话题 /person_detect_status (std_msgs/String) —— 报告串
#   文件 data/reports/report_person.txt + data/evidence/R_*.jpg
# 相机: webcam 848×480 MJPG, 装在云台上; 通过 /usb_cam/image_raw 订阅,
#       无帧时自动拉起 usb_cam_mjpeg.launch (RosCamera)。
# 云台: 只走 gimbal_server 的 /gun/aim /gun/fire /gun/home 服务, 绝不直写串口。
# 用法:
#   rosrun raicom2026 webcam_person_gun.py              # 等 /person_detect 触发
#   rosrun raicom2026 webcam_person_gun.py --zone A     # 启动即扫 A 区后退出
#   rosrun raicom2026 webcam_person_gun.py --dryrun     # 只扫/数/报, 不打靶不转云台
# 标定: param/params.yaml (ppd_h/ppd_v/s_h/s_v/t0/tilt_offset_*), 未标定用默认值。
# ============================================================
import os
import sys
import time
import math
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
#   from raicom2026.srv import GunAim  -> ModuleNotFoundError (实机踩过)。
# devel 生成包的 __init__.py 已用 extend_path 把本包 src 并入 __path__,
# 所以源码模块(yolo_onnx/camera_ros/report/scan_logic)与生成 srv 都能直接解析。

import cv2
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

# cv_bridge 必须 python3 版并提到最前 (RosCamera 内部也会处理, 这里双保险)
sys.path.insert(0, '/home/mowen/cv_bridge_ws/devel/lib/python3/dist-packages')

import rospkg
from raicom2026.yolo_onnx import YOLOv8ONNX, draw_detections
from raicom2026.report import ReportStore, make_person_report, validate_person_line
from raicom2026.scan_logic import (
    HOSTAGE, ENEMY, FRIEND, CLASS_NAMES, clamp,
    px_to_pan, px_to_tilt, distance_from_box_height, tilt_offset,
    Det, cluster_person_detections, choose_aim,
)
from raicom2026.camera_ros import RosCamera, ensure_master
from raicom2026.srv import GunAim, GunFire, DetectTask, DetectTaskResponse
from std_srvs.srv import Trigger

# ============================ 路径 ============================
try:
    PKG = rospkg.RosPack().get_path('raicom2026')
except Exception:
    PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PKG, 'data', 'models', 'person.onnx')
REPORT_DIR = os.path.join(PKG, 'data', 'reports')
EVIDENCE_DIR = os.path.join(PKG, 'data', 'evidence')
PARAMS_PATH = os.path.join(PKG, 'param', 'params.yaml')
PERSON_REPORT_PATH = os.path.join(REPORT_DIR, 'report_person.txt')
MINE_REPORT_PATH = os.path.join(REPORT_DIR, 'report_mine.txt')

# ============================ 相机 ============================
IMAGE_TOPIC = '/usb_cam/image_raw'
WEBCAM_LAUNCH = 'roslaunch raicom2026 usb_cam_mjpeg.launch'
WEBCAM_WAIT = 15
IMG_W_DEF, IMG_H_DEF = 848, 480

# ============================ 模型/置信度 ============================
CONF_SCAN = 0.45         # 扫描时记录检测的最低置信
CONF_CONFIRM_HI = 0.60   # >=此置信无需二帧确认; 在[CONF_SCAN,此)区间补一帧确认
CONF_ENEMY_FIRE = 0.42   # 开火前 enemy 最低置信 (±6° pan 内) (2026-08-13: 0.50 卡死 —
                         # 射击距离上 enemy 框置信在 0.49 附近波动, 略降容忍帧间抖动)
CONF_DANGER = 0.60       # 友军/人质 >=此置信且在 ±pan_ok°×±tilt_ok° 内 -> 跳过
                         # (2026-08-12: 0.50 过低 — 0.5~0.6 的模糊误检框会误触发 SKIP_DANGER
                         #  导致永远无法开火; 真目标置信 0.9+, 提到 0.60 仍保护真目标)
CONF_KNOCKED = 0.40      # 击倒验证: 低于此置信视为已击倒

# ============================ 云台扫描网格 ============================
T0 = 80.0                # 俯仰基准 (标定后写 params.yaml)
TILT_STEP = 10.0         # 两档俯仰差
PAN_POS = [50.0, 75.0, 90.0, 105.0, 130.0]   # 90±40, 步进 25°
PAN_LO, PAN_HI = 5.0, 175.0
TILT_LO, TILT_HI = 70.0, 110.0
SCAN_SETTLE = 0.40       # 舵机写后等稳定再取帧
AIM_SETTLE = 0.15        # 瞄准后额外等稳定 (gimbal_server 每条指令已内置 0.1s)

# ============================ 标定参数 (params.yaml 覆盖) ============================
PPD_H = 12.11            # px/度, 未标定默认 (~70°HFOV / 848px)
PPD_V = 12.11
S_H = 1.0                # 方向符号因子 (标定)
S_V = -1.0               # ⚠️ 默认 -1 (调高 tilt 角 -> 目标在画面中下移, v 增大);
                         #    标定后若实测相反改 +1 (见 param/params.yaml 注释)
                         #    填反了闭环微调会发散, 永远开不了火
TILT_OFFSET_A0 = 0.0     # 弹道垂直补偿 a0 + a1/d (标定后填)
TILT_OFFSET_A1 = 0.0
TARGET_H_M = 0.30        # 贴纸/目标物理高度(m), 用于框高反推距离
DEFAULT_DIST = 1.5       # 距离未知时的补偿用距

# ============================ 时间预算/射击 ============================
ZONE_TIME_BUDGET = 30.0  # 每区人物块硬上限 (2026-08-13: 单位置扫描~3s + 最多3敌军打靶~24s)
SHOOT_COOLDOWN = 1.7     # 节点侧开火间隔 (2026-08-16: 3.2->1.7, 服务端冷却降到1.5s)
SHOTS_PER_TARGET = 4     # 每目标最多射击次数 (2026-08-16: 3->4, 快节奏补射)
FIRE_BURST_MS = 50.0     # 单次点射时长 ms (2026-08-16: 100->50, 每次一枪)
FINE_TUNE_ITER = 3       # 闭环微调最多迭代
CENTER_TOL_DEG = 2.0     # 目标居中判定 (~3% 视场)
VERIFY_PAN_OFFSET = 3.0  # 击倒验证时换的新位姿偏移(pan)

# ============================ 单位置扫描 (2026-08-13) ============================
SCAN_FRAMES = 5          # 单位置连拍帧数 (聚类去重用)
SCAN_FRAME_GAP = 0.3     # 连拍帧间隔 (秒)
FALLBACK_PAN_DEG = 20.0  # 主方向 0 检测时, 左右兜底微扫偏移角 (度)

# ============================ 各区固定云台预偏角 (2026-08-14) ============================
# 服务/话题触发无需再传 gimbal_bias, 按区自动取 (A/B 右转 60°, C/D 左转 60°)
ZONE_BIAS = {'A': -60.0, 'B': -60.0, 'C': 60.0, 'D': 60.0}

# 报告串/触发话题
TRIGGER_TOPIC = '/person_detect'
STATUS_TOPIC = '/person_detect_status'


def load_params(path):
    """极简 key: value 解析 (兼容 calibrate.py 输出), 返回 dict。"""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            raw = raw.split('#')[0].strip()
            if not raw or ':' not in raw:
                continue
            k, v = raw.split(':', 1)
            try:
                d[k.strip()] = float(v.strip())
            except ValueError:
                d[k.strip()] = v.strip()
    return d


class WebcamPersonGun:
    def __init__(self, zone_once=None, dryrun=False):
        self.zone_once = zone_once
        self.dryrun = dryrun
        self.busy = False
        self.busy_lock = threading.Lock()   # --zone 模式与触发回调可能并发, 防双扫
        self._abort = False                 # shutdown 置位, 扫描/瞄准循环据此快速退出
        self.allow_fire = not dryrun        # 标定确认后(下方)会再收紧
        self.last_shot_time = 0.0
        self.zone_start = time.time()
        self.gimbal_ok = False
        self.aim_srv = self.fire_srv = self.home_srv = None
        self.cam = None
        self.img_w, self.img_h = IMG_W_DEF, IMG_H_DEF

        # ---------- 标定参数 ----------
        p = load_params(PARAMS_PATH)
        self.ppd_h = p.get('ppd_h', PPD_H)
        self.ppd_v = p.get('ppd_v', PPD_V)
        self.s_h = p.get('s_h', S_H)
        self.s_v = p.get('s_v', S_V)
        self.t0 = p.get('t0', T0)
        self.tilt_step = p.get('tilt_step', TILT_STEP)
        self.tilt_a0 = p.get('tilt_offset_a0', TILT_OFFSET_A0)
        self.tilt_a1 = p.get('tilt_offset_a1', TILT_OFFSET_A1)
        self.pan_offset = p.get('pan_offset_deg', 0.0)   # 枪管水平安装偏差补偿 (度, 实测标定)
        self.target_h_m = p.get('target_h_m', TARGET_H_M)
        self.focal_px = p.get('focal_px', self.ppd_h * 180.0 / math.pi)
        rospy.loginfo('[person] 标定参数 ppd=%.2f/%.2f s=%.1f/%.1f t0=%.1f '
                      'tilt_offset=%.2f+%.2f/d pan_offset=%.2f°',
                      self.ppd_h, self.ppd_v, self.s_h, self.s_v, self.t0,
                      self.tilt_a0, self.tilt_a1, self.pan_offset)
        # ---------- 标定确认门 (评审#6) ----------
        # 未确认标定(param/params.yaml 缺 'calibrated: true')时禁止开火:
        # ppd/s_v 未标定 -> 瞄准发散打不中, a0/a1=0 -> 无弹道补偿系统性脱靶,
        # 强行开火 = 白丢 10 分还可能把枪口指向友军。宁走计数模式保平安。
        self.calibrated = str(p.get('calibrated', 'false')).strip().lower() in ('true', '1', 'yes')
        if not self.calibrated:
            if dryrun:
                rospy.logwarn('[person] 未确认标定, dryrun 计数模式(不打靶)')
            else:
                rospy.logfatal('[person] 未确认标定(缺 calibrated: true) -> 禁开火, '
                               '仅计数。标定完成并写入 param/params.yaml 后重启')
                self.allow_fire = False

        # ---------- 相机 ----------
        try:
            self.cam = RosCamera(IMAGE_TOPIC, WEBCAM_LAUNCH, wait=WEBCAM_WAIT)
        except Exception as e:
            rospy.logfatal('[person] 相机初始化失败: %s', e)
            self.cam = None
        if self.cam is not None and self.cam.has_frame():
            self.img_w, self.img_h = self.cam.w, self.cam.h
        self.webcam_ok = self.cam is not None and self.cam.has_frame()
        if not self.webcam_ok:
            rospy.logwarn('[person] webcam 无帧: 只能报 0, 不打靶 (TIER-2 降级)')

        # ---------- 模型 ----------
        # 模型缺失/损坏直接失败退出 (评审发现: 相机块有 try/except 降级, 模型块没有)。
        # 故意不静默降级 —— 没有模型人物识别全 0, '静默继续' = 现场看着在跑实际在报空,
        # 比启动即失败更难察觉。启动失败 + 清晰报错是正确行为。
        try:
            self.model = YOLOv8ONNX(MODEL_PATH, conf_thres=CONF_SCAN, iou_thres=0.45,
                                    intra_op_num_threads=2)
            rospy.loginfo('[person] person.onnx 加载完成: %s', MODEL_PATH)
        except Exception as e:
            rospy.logfatal('[person] person.onnx 加载失败: %s (%s) — 检查 data/models/person.onnx',
                           e, MODEL_PATH)
            sys.exit(1)

        # ---------- 云台服务 (可选) ----------
        # 2026-08-15: 不用 wait_for_service (python3 下返回 None 不可靠),
        # 直接绑定 ServiceProxy; 服务晚就绪也能由 _ensure_gimbal 运行期恢复。
        if not dryrun and self.webcam_ok:
            try:
                self.aim_srv = rospy.ServiceProxy('/gun/aim', GunAim)
                self.fire_srv = rospy.ServiceProxy('/gun/fire', GunFire)
                self.home_srv = rospy.ServiceProxy('/gun/home', Trigger)
                self.gimbal_ok = True
            except Exception:
                self.gimbal_ok = False
        if not self.gimbal_ok:
            rospy.logwarn('[person] 云台服务暂未绑定 (运行期会自动重试)')

        # ---------- 报告/证据 ----------
        self.store = ReportStore(PERSON_REPORT_PATH, MINE_REPORT_PATH)
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        self.status_pub = rospy.Publisher(STATUS_TOPIC, String, queue_size=5)
        rospy.Subscriber(TRIGGER_TOPIC, String, self.cb_trigger, queue_size=10)
        # 方案D: 服务触发 (nav 主路径) —— 同步一问一答, 报告串直接返回
        self.srv = rospy.Service('/person_detect_service', DetectTask,
                                 self.srv_detect_cb)

        # ---------- 调试视图 (2026-08-13: 实时看识别/瞄准画面) ----------
        # ~debug_view:=true 时发布带检测框+瞄准状态文字的图像到 /person/debug_image
        # 终端查看: rosrun image_view image_view image:=/person/debug_image
        self.debug_view = rospy.get_param('~debug_view', True)
        self.debug_pub = None
        self._dbg_last_boxes = ([], [], [])
        if self.debug_view:
            try:
                from cv_bridge import CvBridge as _Br
                self._dbg_bridge = _Br()
                self.debug_pub = rospy.Publisher('/person/debug_image', Image, queue_size=1)
                # 空闲时也持续发布画面 (5Hz), 任务中由 _publish_debug 接管
                rospy.Timer(rospy.Duration(0.2), self._debug_tick)
                rospy.loginfo('[person] 调试视图已启用: /person/debug_image (空闲时 5Hz)')
            except Exception as e:
                rospy.logwarn('[person] 调试视图初始化失败: %s', e)
                self.debug_view = False

    # ---------------- 云台动作 ----------------
    def _ensure_gimbal(self):
        """运行期云台服务重试 (2026-08-13): 启动竞态下 gimbal 服务可能晚于
        wait_for_service(3s) 注册。每次任务前探测一次, 恢复了就重新绑定服务代理。
        2026-08-15: webcam_ok 实时重评估 + 不用 wait_for_service (python3 下
        返回 None 不可靠, 导致永远恢复不了云台服务 -> 云台不转/不打靶)。
        改为直接绑定 ServiceProxy, 由 aim 调用结果判定服务是否可用。"""
        if self.gimbal_ok:
            return True
        if self.dryrun:
            return False
        self.webcam_ok = self.cam is not None and self.cam.has_frame()
        if not self.webcam_ok:
            return False
        try:
            self.aim_srv = rospy.ServiceProxy('/gun/aim', GunAim)
            self.fire_srv = rospy.ServiceProxy('/gun/fire', GunFire)
            self.home_srv = rospy.ServiceProxy('/gun/home', Trigger)
            self.gimbal_ok = True
            rospy.loginfo('[person] 云台服务已绑定 (运行期延迟恢复)')
        except Exception as e:
            rospy.logwarn_throttle(5.0, '[person] 云台服务绑定失败: %s', e)
        return self.gimbal_ok

    def aim(self, pan, tilt):
        pan = clamp(pan, PAN_LO, PAN_HI)
        tilt = clamp(tilt, TILT_LO, TILT_HI)
        if not self.gimbal_ok:
            return False
        try:
            resp = self.aim_srv(pan, tilt)
            time.sleep(AIM_SETTLE)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(2.0, '[person] /gun/aim 异常: %s', e)
            return False

    def home(self):
        if not self.gimbal_ok:
            return
        try:
            self.home_srv()
        except rospy.ServiceException:
            pass

    def _fresh_frame_safe(self):
        """开火前用最新一帧再过一次安全门 (防冷却睡眠期间友军/人质走入射程)。
        返回 True=可以开火。无帧也视为不安全(不贸然扣扳机)。
        2026-08-13 修复: 开火前已做 pan_offset 补偿转动(画面中心≠枪管指向),
        安全门必须按枪管指向 ±pan_ok 判定友军/人质, 而不是画面中心。
        否则补偿后友军恰好落在画面中心附近 -> 永远 SKIP_DANGER 打不了敌。"""
        frame = self._wait_frame(timeout=0.8)
        if frame is None:
            return False
        boxes, scores, ids = self.model.detect(frame)
        dets = [(c, float(s), b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
                for b, s, c in zip(boxes, scores, ids)]
        # 枪管指向 = 画面中心角 - pan_offset (pan_offset = -枪管安装偏差δ)
        # 目标相对枪管角距 = s_h*(u-W/2)/ppd_h + pan_offset
        # => 像素等效: u' = u + s_h*ppd_h*pan_offset (2026-08-14 符号修正)
        hx_shift = -self.s_h * self.ppd_h * self.pan_offset
        dets_shift = [(c, s, u - hx_shift, v) for (c, s, u, v) in dets]
        action, _, _ = choose_aim(
            dets_shift, self.img_w, self.img_h, self.ppd_h, self.ppd_v,
            self.s_h, self.s_v,
            enemy_conf_min=CONF_ENEMY_FIRE, danger_conf_min=CONF_DANGER,
            center_tol=CENTER_TOL_DEG)
        return action != 'SKIP_DANGER'

    def _fire_once(self):
        """扣一次扳机 (含冷却等待 + 开火前安全门复核)。
        返回 'FIRED' / 'DANGER' / 'COOLDOWN_REJ' / 'SERVICE_ERR' / 'ABORT'。"""
        if self._abort:
            return 'ABORT'
        wait = SHOOT_COOLDOWN - (time.time() - self.last_shot_time)
        if wait > 0:
            # 冷却睡眠不超过剩余区预算, 防止单区拖超 20s 顺延后续区
            remain = ZONE_TIME_BUDGET - (time.time() - self.zone_start)
            time.sleep(max(0.0, min(wait, remain)))
            if self._abort:                # 冷却后、扣扳机前再确认没被中断
                return 'ABORT'
        if not self._fresh_frame_safe():
            return 'DANGER'
        if self._abort:                    # 开火前最终确认 (防 shutdown 竞态)
            return 'ABORT'
        try:
            resp = self.fire_srv(FIRE_BURST_MS)
            if resp.success:
                self.last_shot_time = time.time()
                return 'FIRED'
            rospy.logwarn('[person] /gun/fire 拒绝: %s', resp.message)
            return 'COOLDOWN_REJ'
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(2.0, '[person] /gun/fire 异常: %s', e)
            return 'SERVICE_ERR'

    def _wait_frame(self, timeout=1.0):
        """取一帧较新的图像 (等待相机出帧)。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            f = self.cam.get_frame() if self.cam else None
            if f is not None:
                return f
            time.sleep(0.03)
        return None

    # ---------------- 调试视图 ----------------
    def _debug_tick(self, event):
        """空闲时 5Hz 发布实时画面 (任务执行时由 _publish_debug 接管, 避免重复)。"""
        if not self.debug_view or self.debug_pub is None:
            return
        if self.busy:
            return
        frame = self.cam.get_frame() if self.cam else None
        if frame is None:
            return
        self._publish_debug(frame, 'IDLE 等待触发\n'
                                   '服务: rosservice call /person_detect_service "{zone: A, gimbal_bias: -60.0}"\n'
                                   '话题: rostopic pub -1 /person_detect std_msgs/String "A"')

    def _publish_debug(self, frame, text, boxes=None, scores=None, ids=None):
        """发布带标注的调试图像 (检测框 + 状态文字) 到 /person/debug_image。"""
        if not self.debug_view or self.debug_pub is None or frame is None:
            return
        try:
            img = frame.copy()
            if boxes:
                img = draw_detections(img, boxes, scores or [], ids or [], CLASS_NAMES)
            # 状态文字分多行画在顶部
            lines = text.split('\n')
            for i, ln in enumerate(lines):
                cv2.putText(img, ln, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 255), 2, cv2.LINE_AA)
            self.debug_pub.publish(self._dbg_bridge.cv2_to_imgmsg(img, 'bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(5.0, '[person] 调试图发布失败: %s', e)

    # ---------------- 单位置扫描 (2026-08-13) ----------------
    def _scan_at(self, pan_cmd, tilt_cmd, n_frames):
        """在指定云台位置连拍 n 帧, 收集检测 (映射绝对角)。返回 Det 列表。"""
        dets = []
        if self.gimbal_ok:
            if not self.aim(pan_cmd, tilt_cmd):
                return dets
        for i in range(n_frames):
            if self._abort:
                break
            time.sleep(SCAN_FRAME_GAP if i > 0 else 0.0)
            frame = self._wait_frame(timeout=0.8)
            if frame is None:
                continue
            boxes, scores, ids = self.model.detect(frame)
            for b, s, c in zip(boxes, scores, ids):
                if s < CONF_SCAN:
                    continue
                u, v = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
                ap, at = self._abs_angle(pan_cmd, tilt_cmd, u, v)
                dets.append(Det(c, float(s), u, v, b[2], b[3], ap, at, tilt_cmd,
                                frame, boxes, scores, ids))
            # 调试视图
            self._publish_debug(frame,
                                'ZONE scan pan=%.0f tilt=%.0f frame=%d/%d det=%d' % (
                                    pan_cmd, tilt_cmd, i + 1, n_frames, len(boxes)),
                                boxes, scores, ids)
        return dets

    def scan_single(self, gimbal_bias):
        """单位置识别: 云台转到 (90+gimbal_bias, t0) 连拍 SCAN_FRAMES 帧。"""
        pan = clamp(90.0 + gimbal_bias, PAN_LO, PAN_HI)
        tilt = clamp(self.t0, TILT_LO, TILT_HI)
        rospy.loginfo('[person] 单位置扫描: bias=%.1f -> pan=%.1f tilt=%.1f, %d帧',
                      gimbal_bias, pan, tilt, SCAN_FRAMES)
        return self._scan_at(pan, tilt, SCAN_FRAMES)

    def scan_fallback(self, gimbal_bias):
        """兜底微扫: 主方向 0 检测时, 左右 ±FALLBACK_PAN_DEG 各拍 2 帧 (防航点误差)。"""
        rospy.logwarn('[person] 主方向无检测, 兜底微扫 ±%.0f°', FALLBACK_PAN_DEG)
        dets = []
        tilt = clamp(self.t0, TILT_LO, TILT_HI)
        for dp in (+FALLBACK_PAN_DEG, -FALLBACK_PAN_DEG):
            if self._abort:
                break
            pan = clamp(90.0 + gimbal_bias + dp, PAN_LO, PAN_HI)
            dets.extend(self._scan_at(pan, tilt, 2))
        return dets

    # ---------------- 扫描 (旧 10 视区, 保留备用) ----------------
    def _abs_angle(self, cmd_pan, cmd_tilt, u, v):
        """像素中心 → 绝对云台角 (钳位)。"""
        ap = clamp(px_to_pan(cmd_pan, u, self.img_w, self.ppd_h, self.s_h),
                   PAN_LO, PAN_HI)
        at = clamp(px_to_tilt(cmd_tilt, v, self.img_h, self.ppd_v, self.s_v),
                   TILT_LO, TILT_HI)
        return ap, at

    def scan_grid(self):
        """两档俯仰 × 5 位 pan 扫描, 收集 Det 列表 (有云台时用)。
        Det.level 记录所属俯仰档(命令角), 供聚类跨档像素连续性去重。"""
        dets = []
        levels = [clamp(self.t0, TILT_LO, TILT_HI),
                  clamp(self.t0 + self.tilt_step, TILT_LO, TILT_HI)]
        for tilt in levels:
            if self._abort:
                break
            for pan in PAN_POS:
                if self._abort:
                    break
                pan = clamp(pan, PAN_LO, PAN_HI)
                if not self.aim(pan, tilt):
                    continue
                time.sleep(SCAN_SETTLE)          # 舵机稳定后再取帧
                frame = self._wait_frame(timeout=0.8)
                if frame is None:
                    continue
                boxes, scores, ids = self.model.detect(frame)
                for b, s, c in zip(boxes, scores, ids):
                    if s < CONF_SCAN:
                        continue
                    u, v = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
                    ap, at = self._abs_angle(pan, tilt, u, v)
                    det = Det(c, float(s), u, v, b[2], b[3], ap, at, tilt,
                              frame, boxes, scores, ids)
                    dets.append(det)
                    # 中置信补一帧确认 (方案 §3.1)
                    if CONF_SCAN <= s < CONF_CONFIRM_HI:
                        time.sleep(0.12)
                        f2 = self._wait_frame(timeout=0.5)
                        if f2 is None:
                            continue
                        b2, s2, c2 = self.model.detect(f2)
                        matched = False
                        for bx, sx, cx in zip(b2, s2, c2):
                            if cx == c and abs((bx[0] + bx[2] / 2.0) - u) < b[2]:
                                matched = True
                                if sx > s:       # 确认帧置信更高才升级, 否则保留原检测
                                    det.conf = float(sx)
                                    det.u, det.v = bx[0] + bx[2] / 2.0, bx[1] + bx[3] / 2.0
                                    ap, at = self._abs_angle(pan, tilt, det.u, det.v)
                                    det.pan, det.tilt = ap, at
                                break
                        if not matched:
                            dets.remove(det)     # 确认帧无同位置同类 -> 单帧误检, 不计数
        return dets

    def scan_static(self, dur=1.5):
        """无云台/干跑: 静止相机按"类别+像素中心"去重计数。
        相机不动, 角度映射无意义(绝对角统一置 0), 只保证同一视角下不重复计。"""
        kept = []
        t0 = time.time()
        while time.time() - t0 < dur and not self._abort:
            frame = self._wait_frame(timeout=0.5)
            if frame is None:
                continue
            boxes, scores, ids = self.model.detect(frame)
            for b, s, c in zip(boxes, scores, ids):
                if s < CONF_SCAN:
                    continue
                u, v = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
                hit = None
                for det in kept:
                    if (det.cls == c and abs(det.u - u) < max(40.0, b[2] * 0.8)
                            and abs(det.v - v) < max(60.0, b[3] * 1.0)):
                        hit = det
                        break
                if hit is None:
                    kept.append(Det(c, float(s), u, v, b[2], b[3], 0.0, 0.0, 0.0,
                                    frame, boxes, scores, ids))
                elif s > hit.conf:
                    hit.conf = float(s)
                    hit.u, hit.v, hit.w, hit.h = u, v, b[2], b[3]
                    hit.frame, hit.boxes = frame, boxes
                    hit.scores, hit.ids = scores, ids
        return kept

    def _scan(self):
        """按云台可用性分发扫描。"""
        if self.gimbal_ok:
            return self.scan_grid()
        return self.scan_static()

    # ---------------- 打靶 ----------------
    def aim_and_fire(self, cluster):
        """对单个 enemy 瞄准→安全门→闭环微调→开火。
        返回 'FIRED' / 'SKIP_DANGER' / 'NO_ENEMY' / 'NO_FRAME' / 'GIMBAL_DOWN'
              / 'NOT_CENTERED' / 'ABORT'。"""
        pan = clamp(cluster.pan, PAN_LO, PAN_HI)
        tilt = clamp(cluster.tilt, TILT_LO, TILT_HI)
        # 弹道垂直补偿 (按距离查表)
        d = distance_from_box_height(self.target_h_m, self.focal_px,
                                     cluster.best.h) or DEFAULT_DIST
        tilt += tilt_offset(d, self.tilt_a0, self.tilt_a1)
        tilt = clamp(tilt, TILT_LO, TILT_HI)

        for _ in range(FINE_TUNE_ITER + 1):
            if self._abort:
                return 'ABORT'
            if not self.gimbal_ok:
                return 'GIMBAL_DOWN'
            if not self.aim(pan, tilt):
                return 'GIMBAL_DOWN'
            frame = self._wait_frame(timeout=1.0)
            if frame is None:
                return 'NO_FRAME'
            boxes, scores, ids = self.model.detect(frame)
            dets = [(c, float(s), b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
                    for b, s, c in zip(boxes, scores, ids)]
            action, dp, dt = choose_aim(
                dets, self.img_w, self.img_h, self.ppd_h, self.ppd_v,
                self.s_h, self.s_v,
                enemy_conf_min=CONF_ENEMY_FIRE, danger_conf_min=CONF_DANGER,
                center_tol=CENTER_TOL_DEG)
            # 调试视图: 瞄准过程实时显示判定结果
            self._publish_debug(frame, 'AIM pan=%.1f tilt=%.1f -> %s (dp=%.1f dt=%.1f)' % (
                pan, tilt, action, dp, dt), boxes, scores, ids)
            if action == 'SKIP_DANGER':
                return 'SKIP_DANGER'
            if action == 'NO_ENEMY':
                return 'NO_ENEMY'
            if action == 'NOTHING':
                return 'NO_ENEMY'
            if action == 'FIRE':
                # 枪管水平安装偏差补偿 (pan_offset_deg 实测标定):
                # 相机闭环微调已对准 enemy, 开火前平移 pan 使枪管指向目标
                # (补偿放在最后一步, 避免被下一轮闭环微调抵消)
                final_pan = clamp(pan + self.pan_offset, PAN_LO, PAN_HI)
                if abs(final_pan - pan) > 0.05:
                    if not self.aim(final_pan, tilt):
                        return 'GIMBAL_DOWN'
                res = self._fire_once()
                if res == 'FIRED':
                    return 'FIRED'
                if res == 'ABORT':
                    return 'ABORT'
                if res == 'DANGER':
                    return 'SKIP_DANGER'
                return 'COOLDOWN'
            # ADJUST: 按修正角再瞄准
            pan = clamp(pan + dp, PAN_LO, PAN_HI)
            tilt = clamp(tilt + dt, TILT_LO, TILT_HI)
        return 'NOT_CENTERED'

    def _verify_knocked(self, cluster):
        """开火后换新位姿验证目标已不在 (conf<0.40 视为击倒)。
        只检查目标原位置附近 (中心 ±0.2 幅宽) 的 enemy, 避免邻近第二个敌人
        让验证永远失败、白耗射击。"""
        time.sleep(0.4)
        pan = clamp(cluster.pan + VERIFY_PAN_OFFSET
                    if cluster.pan + VERIFY_PAN_OFFSET <= PAN_HI
                    else cluster.pan - VERIFY_PAN_OFFSET,
                    PAN_LO, PAN_HI)
        tilt = clamp(cluster.tilt, TILT_LO, TILT_HI)
        if not self.gimbal_ok:
            return False
        if not self.aim(pan, tilt):
            return False
        frame = self._wait_frame(timeout=1.0)
        if frame is None:
            return False
        boxes, scores, ids = self.model.detect(frame)
        hx, hy = self.img_w / 2.0, self.img_h / 2.0
        for b, s, c in zip(boxes, scores, ids):
            if c == ENEMY and s >= CONF_KNOCKED:
                ub, vb = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
                if abs(ub - hx) < 0.2 * self.img_w and abs(vb - hy) < 0.2 * self.img_h:
                    return False          # 目标位置附近仍有 enemy -> 未击倒
        return True

    def _target_still_there(self):
        """轻量验证: 当前画面中心附近是否还有 enemy (不打云台、不换位姿)。
        用于连发间提前停火 (目标已倒就不浪费子弹/时间)。无帧视为还在(继续打)。"""
        frame = self._wait_frame(timeout=0.6)
        if frame is None:
            return True
        boxes, scores, ids = self.model.detect(frame)
        hx, hy = self.img_w / 2.0, self.img_h / 2.0
        for b, s, c in zip(boxes, scores, ids):
            if c == ENEMY and s >= CONF_KNOCKED:
                ub, vb = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
                if abs(ub - hx) < 0.2 * self.img_w and abs(vb - hy) < 0.2 * self.img_h:
                    return True
        return False

    def shoot_cluster(self, cluster):
        """对单个 enemy 目标射击, 返回 (knocked, shots)。
        2026-08-14: 第1发完整瞄准; 后续发同位置直射(不重瞄), 每发后轻量
        检查目标是否已倒, 倒了提前停; 打满 SHOTS_PER_TARGET 后统一验证。"""
        shots = 0
        for i in range(SHOTS_PER_TARGET):
            if self._abort:
                break
            if time.time() - self.zone_start > ZONE_TIME_BUDGET:
                break
            if i == 0:
                res = self.aim_and_fire(cluster)
                if res == 'FIRED':
                    shots += 1
                elif res == 'SKIP_DANGER':
                    rospy.logwarn('[person] 友军/人质在瞄准区内, 放弃射击(不赌扣10)')
                    return False, shots
                elif res == 'NO_ENEMY':
                    continue
                else:  # NO_FRAME/GIMBAL_DOWN/NOT_CENTERED/COOLDOWN
                    break
            else:
                res = self._fire_once()
                if res == 'FIRED':
                    shots += 1
                elif res == 'DANGER':
                    # 安全门触发: 重新瞄准复核一次 (可能是检测抖动)
                    res2 = self.aim_and_fire(cluster)
                    if res2 == 'FIRED':
                        shots += 1
                    elif res2 == 'SKIP_DANGER':
                        rospy.logwarn('[person] 友军/人质在瞄准区内, 放弃射击(不赌扣10)')
                        return False, shots
                    else:
                        break
                else:  # ABORT/COOLDOWN_REJ/SERVICE_ERR
                    break
            if shots > 0 and not self._target_still_there():
                rospy.loginfo('[person] 目标已消失, 提前停止射击')
                return True, shots
        # 打满后统一验证击倒
        if shots > 0:
            if self._verify_knocked(cluster):
                return True, shots
        return False, shots

    # ---------------- 证据图 ----------------
    def save_evidence(self, zone, clusters):
        """单张证据图 (2026-08-14): 取检测框最多的一帧, 标出全部人偶。
        调用时机在打靶之前 (run_zone 内), 保证画面是未射击状态。"""
        best = None
        for c in clusters:
            d = c.best
            if d.frame is None:
                continue
            if best is None or len(d.boxes) > len(best[1]):
                best = (d.frame, d.boxes, d.scores, d.ids)
        if best is None:
            return
        name = 'R_%s.jpg' % zone
        img = draw_detections(best[0], best[1], best[2], best[3], CLASS_NAMES)
        path = os.path.join(EVIDENCE_DIR, name)
        cv2.imwrite(path, img)
        rospy.loginfo('[person] 证据图 %s', path)

    # ---------------- 主流程 ----------------
    def run_zone(self, zone, gimbal_bias=0.0):
        zone = zone.strip().upper()
        if zone not in 'ABCD':
            rospy.logwarn('[person] 非法区名 %r, 忽略', zone)
            return None
        # 2026-08-14: 按区固定云台预偏角, 忽略调用方传入的 gimbal_bias
        gimbal_bias = ZONE_BIAS.get(zone, 0.0)
        with self.busy_lock:
            if self.busy:
                rospy.logwarn('[person] 正在扫描中, 忽略新触发 %s', zone)
                return None
            self.busy = True
        self.zone_start = time.time()
        fire_results = []
        try:
            if self._abort:
                return None
            self._ensure_gimbal()   # 运行期重试云台服务 (启动竞态兜底)
            # 2026-08-13: 单位置识别 (不再扫 10 视区) —— 云台直接转到预设方向
            dets = self.scan_single(gimbal_bias)
            if not dets:
                # 主方向无检测 -> 兜底 ±20° 微扫 (防航点/偏角误差导致漏检)
                dets = self.scan_fallback(gimbal_bias)
            clusters = cluster_person_detections(dets,
                                                  tilt_step=self.tilt_step,
                                                  ppd_v=self.ppd_v)

            enemy = sum(1 for c in clusters if c.cls == ENEMY)
            friend = sum(1 for c in clusters if c.cls == FRIEND)
            hostage = sum(1 for c in clusters if c.cls == HOSTAGE)
            rospy.loginfo('[person] %s区 扫描完成: 检测到 %d 目标 (敌=%d 友=%d 质=%d)',
                          zone, len(clusters), enemy, friend, hostage)

            # 敌军 0~3 个均合法 (2026-08-13): 敌军 0 时只上报不打靶
            report = make_person_report(zone, enemy, friend, hostage)
            self.store.add_person(zone, enemy, friend, hostage)   # 先把分拿到
            self.status_pub.publish(report)
            rospy.loginfo('[person] === %s ===', report)

            self.save_evidence(zone, clusters)

            # 打靶: 只打 enemy, 逐个射击 (多个敌军逐个击倒, 安全门在 choose_aim 内)
            if self.gimbal_ok and self.allow_fire and enemy > 0:
                rospy.loginfo('[person] %s区 打靶开始: %d 个敌军目标', zone, enemy)
                for idx, c in enumerate(sorted([c for c in clusters if c.cls == ENEMY],
                                               key=lambda x: -x.conf)):
                    if self._abort:
                        break
                    if time.time() - self.zone_start > ZONE_TIME_BUDGET:
                        rospy.logwarn('[person] %s区 预算 %.0fs 耗尽, 停止打靶',
                                      zone, ZONE_TIME_BUDGET)
                        break
                    rospy.loginfo('[person] 瞄准 enemy#%d pan=%.1f conf=%.2f ...',
                                  idx + 1, c.pan, c.conf)
                    knocked, shots = self.shoot_cluster(c)
                    rospy.loginfo('[person] %s区 enemy#%d pan=%.1f: 击倒=%s 射%d发 (耗时%.1fs)',
                                  zone, idx + 1, c.pan, knocked, shots,
                                  time.time() - self.zone_start)
                    fire_results.append('敌#%d %s(%d发)' % (
                        idx + 1, '击倒' if knocked else '未倒', shots))
            elif enemy > 0:
                rospy.logwarn('[person] %s区 存在敌军但未开火 (allow_fire=%s gimbal_ok=%s)',
                              zone, self.allow_fire, self.gimbal_ok)
            if self.gimbal_ok:
                self.home()
            rospy.loginfo('[person] %s区 完成, 总耗时 %.1fs', zone, time.time() - self.zone_start)
            # 2026-08-14: 排版结果打印到 stdout (start_person.sh 终端 1 实时显示)
            print('')
            print('=' * 52)
            print('             %s 区  识别结果' % zone)
            print('-' * 52)
            print('   敌军: %d     友军: %d     人质: %d' % (enemy, friend, hostage))
            if fire_results:
                print('   打靶: %s' % ' | '.join(fire_results))
            print('=' * 52)
            print('')
            return report
        finally:
            self.busy = False

    def cb_trigger(self, msg):
        try:
            self.run_zone(msg.data)
        except Exception as e:
            rospy.logerr('[person] 扫描异常: %s', e)
            self.busy = False

    def srv_detect_cb(self, req):
        """方案D: /person_detect_service 服务回调 (rospy.Service 独立线程执行,
        不阻塞图像订阅回调)。run_zone 内部有 busy_lock 防并发, busy 时返回 None。
        req.gimbal_bias: 云台预偏角 (A/B 点右侧 -60, C/D 点左侧 +60, 由 nav 航点配置)。"""
        zone = (req.zone or '').strip().upper()
        if zone not in 'ABCD':
            rospy.logwarn('[person] 服务收到非法区名 %r', req.zone)
            return DetectTaskResponse(False, '', 'bad_zone:%s' % req.zone)
        try:
            report = self.run_zone(zone, gimbal_bias=float(getattr(req, 'gimbal_bias', 0.0)))
        except Exception as e:
            rospy.logerr('[person] 服务任务异常: %s', e)
            return DetectTaskResponse(False, '', 'error:%s' % e)
        if report:
            return DetectTaskResponse(True, report, 'ok')
        # run_zone 返回 None = busy 或 abort
        return DetectTaskResponse(False, '', 'busy_or_abort')

    def shutdown(self):
        # 评审#12: 扫描在订阅回调线程里跑, SIGTERM 只中断主线程 spin。
        # 先置 _abort 让扫描/瞄准/开火循环立即退出, 再归位/关相机,
        # 避免 shutdown 进行中回调线程还在发 /gun/fire。
        self._abort = True
        try:
            if self.gimbal_ok:
                self.home()
        except Exception:
            pass
        if self.cam is not None:
            self.cam.shutdown()


def _sig_to_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description='人物识别+打靶节点')
    parser.add_argument('--zone', default=None, help='启动即扫指定区并退出 (A~D)')
    parser.add_argument('--dryrun', action='store_true',
                        help='只扫/数/报, 不转云台不打靶')
    args = parser.parse_args(rospy.myargv(sys.argv)[1:])   # 剥掉 __name:=/__log:= 等 ROS 参数

    ensure_master()   # 必须在 init_node 之前: 无 master 时 init 会直接崩
    rospy.init_node('webcam_person_gun', anonymous=False)
    # rospy.init_node 会覆盖 SIGTERM/SIGINT, 这里重新注册 (collect.py 已验证的模式)
    signal.signal(signal.SIGTERM, _sig_to_interrupt)
    signal.signal(signal.SIGINT, _sig_to_interrupt)

    node = WebcamPersonGun(zone_once=args.zone, dryrun=args.dryrun)
    try:
        if args.zone:
            node.run_zone(args.zone)
        else:
            rospy.loginfo('[person] 就绪, 等待 /person_detect (A~D) 触发')
            rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rospy.loginfo('[person] 退出')


if __name__ == '__main__':
    main()
