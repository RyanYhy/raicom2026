#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# nav_competition.py —— RAICOM2026 自主导航编排节点 (本包新版)
# ------------------------------------------------------------
# 职责: 整场比赛流程编排 ——
#   BOOT:   等 /map + 连接 move_base + 发布 AMCL 初始位姿
#   NAV:    move_base action 巡航到各区航点 (actionlib)
#   TASK:   触发 /person_detect|/mine_detect 并等字节精确状态串
#   REPORT: 调用 report.assemble_report 合并 report.txt (恰好 6 行)
# 替代 1205/main/nav_competition.py (旧版, 只读参考)。与旧版关键差异:
#   * 绝不直写 /dev/gun (C5) —— 云台/打靶全在 webcam_person_gun.py
#   * 触发契约是新节点: /person_detect "A"~"D" (旧版发 "start" 会非法)
#   * 航点读 param/nav_competition.yaml (实机 8/10 重测后只改配置不改代码)
#   * 新增 mine_detect、AMCL 初始位姿、每段导航超时+重试、全局预算、
#     Ⅳ 走廊横向绕行兜底、状态话题 /nav_competition/state、收尾合并报告
# 只通过 action / 话题协作, 不 import tf2_ros (py3 下不可用)。
#
# 用法:
#   roslaunch raicom2026 nav_competition_2026.launch use_rviz:=false
#   rosrun raicom2026 nav_competition.py --print-course   # 只打印解析后的航点课程
# ============================================================
import argparse
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import yaml

import rospy

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# py3 下不能 sys.path.insert(0, ../src) —— 会 shadow 掉 devel 里 catkin 生成的
# raicom2026 包 (含 srv/msg 生成模块)。devel 生成包的 __init__.py 已用
# extend_path 把源码并入, 所以源码模块 report/nav_logic 能直接解析 (同 gimbal_server)。
from raicom2026.nav_logic import (
    yaw_deg_to_quat, validate_course, lateral_offset_goals, Budget,
)
from raicom2026.report import assemble_report
from raicom2026.srv import DetectTask
from std_srvs.srv import SetBool

STATE_TOPIC = '/nav_competition/state'
ZONE_TOPIC = '/nav_competition/zone'
PERSON_SERVICE = '/person_detect_service'
MINE_SERVICE = '/mine_detect_service'
AT_SERVICE = '/antitank_check_service'
AVOID_TOGGLE_SERVICE = '/visual_mine/set_enabled'
AVOID_STATUS_TOPIC = '/visual_mine/status'
INITIAL_POSE_TOPIC = '/initialpose'
MAP_TOPIC = '/map'
YAW_ALIGN_TOL_DEG = 6.0   # 任务点朝向对齐阈值 (2026-08-16: 偏差>6° 原地转向对齐)

GOAL_STATUS_STR = {
    0: 'PENDING', 1: 'ACTIVE', 2: 'PREEMPTED', 3: 'SUCCEEDED',
    4: 'ABORTED', 5: 'REJECTED', 6: 'PREEMPTING', 7: 'RECALLING',
    8: 'RECALLED', 9: 'LOST',
}


def _master_alive():
    try:
        uri = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
        # 兼容 py3.8 的 urllib.parse (无 urllib2)
        if sys.version_info >= (3,):
            from urllib.parse import urlsplit
        else:
            from urlparse import urlsplit
        parsed = urlsplit(uri)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 11311
        s = socket.create_connection((host, int(port)), timeout=1.0)
        s.close()
        return True
    except Exception:
        return False


def _angle_diff(a, b):
    """角度差 a-b 归一化到 [-pi, pi] (朝向对齐用)。"""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def ensure_master():
    """无 master 先拉起 roscore。必须在 init_node 前调用。复刻 camera_ros.ensure_master
    (不 import camera_ros 以免拖入 cv_bridge 依赖 —— 本节点不做图像)。"""
    if _master_alive():
        return True
    rospy.logwarn('[nav] ROS master 未运行, 自动拉起 roscore ...')
    try:
        subprocess.Popen(['roscore'], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 10 and not _master_alive():
            time.sleep(0.2)
    except Exception as e:
        rospy.logwarn('[nav] 自动拉起 roscore 失败: %s', e)
    return _master_alive()


class CompetitionNav:
    def __init__(self):
        rospy.init_node('nav_competition', anonymous=False)

        # ---- 参数 (param/nav_competition.yaml, 加载进私有命名空间 ~) ----
        self.course = rospy.get_param('~course', [])
        try:
            validate_course(self.course)
        except ValueError as e:
            rospy.logfatal('[nav] 航点课程非法: %s', e)
            raise
        self.ip = rospy.get_param('~amcl_initial_pose', {})
        self.person_timeout_s = rospy.get_param('~person_timeout_s', 25)
        self.mine_timeout_s = rospy.get_param('~mine_timeout_s', 12)
        self.nav_timeout_s = rospy.get_param('~nav_timeout_s', 45)
        self.nav_retry = int(rospy.get_param('~nav_retry', 1))
        self.global_budget_s = float(rospy.get_param('~global_budget_s', 280))
        self.iv_lateral_m = float(rospy.get_param('~iv_lateral_offset_m', 0.35))
        self.iv_stall_wait_s = float(rospy.get_param('~iv_stall_wait_s', 3.0))
        self.map_wait_s = float(rospy.get_param('~map_wait_s', 15.0))

        # ---- 话题 ----
        self.state_pub = rospy.Publisher(STATE_TOPIC, String, queue_size=5)
        self.zone_pub = rospy.Publisher(ZONE_TOPIC, String, queue_size=5)
        self.initpose_pub = rospy.Publisher(INITIAL_POSE_TOPIC, PoseWithCovarianceStamped, queue_size=5)

        # 触发等待状态 (方案D: 服务同步调用, 不再需要异步轮询状态)
        self._avoid_status = None    # 仅用于 Ⅳ 走廊避障监控日志
        rospy.Subscriber(AVOID_STATUS_TOPIC, String, self._cb_avoid_status)
        # 2026-08-16: 订阅 amcl_pose, 用于"车已到点但 move_base 判未达"的
        # 实际距离兜底判定 (场地特征稀疏, AMCL 定位有偏差时 move_base 判 ABORTED/超时)
        self._amcl_xy = None
        self._amcl_yaw = None
        self._pose_src = None   # 位姿来源: 'amcl' / 'cartographer' (2026-08-16)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self._cb_amcl_pose)
        # 2026-08-16: Cartographer 定位模式没有 /amcl_pose (那是 AMCL 发布的),
        # 加订阅 /tracked_pose 回退, 否则 _close_to/_align_yaw/_recover 全失效
        rospy.Subscriber('/tracked_pose', PoseStamped, self._cb_tracked_pose)
        # 2026-08-16: 航点箭头 marker (rviz 可视化当前目标航点)
        self.goal_marker_pub = rospy.Publisher('/nav_competition/goal_marker',
                                               Marker, queue_size=1)

        self._budget = Budget(self.global_budget_s)
        self._last_reached = None    # 上一个成功到达的航点 (Ⅳ 横向绕行用)
        self._state = 'INIT'
        self._set_state('BOOT')
        rospy.loginfo('[nav] 航点课程 %d 站: %s',
                      len(self.course), [w['name'] for w in self.course])

    # ---------------- 状态发布 ----------------
    def _set_state(self, s):
        self._state = s
        msg = String('t=%6.1fs %s' % (self._budget.elapsed(), s))
        self.state_pub.publish(msg)
        rospy.loginfo('[nav] [%s]', msg.data)

    # ---------------- 状态串回调 ----------------
    def _cb_avoid_status(self, msg):
        self._avoid_status = msg.data

    def _cb_amcl_pose(self, msg):
        self._pose_src = 'amcl'
        self._amcl_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self._amcl_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _cb_tracked_pose(self, msg):
        """Cartographer 定位位姿回退 (tracking_frame=base_link 在 map 系)。
        base_link 与 base_footprint XY 一致 (仅 z 差), 直接当车位姿用。"""
        self._pose_src = 'cartographer'
        self._amcl_xy = (msg.pose.position.x, msg.pose.position.y)
        q = msg.pose.orientation
        self._amcl_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _close_to(self, x, y, tol):
        """当前 AMCL 定位与实际目标点距离 < tol 米。定位偏差时兜底判定到达。"""
        if self._amcl_xy is None:
            return False
        dx = self._amcl_xy[0] - x
        dy = self._amcl_xy[1] - y
        return (dx * dx + dy * dy) < (tol * tol)

    def _publish_pose(self, x, y, yaw_deg, cov=0.5):
        """发布 initialpose (通用, 初始定位/恢复重定位共用)。"""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = rospy.Time.now()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        qx, qy, qz, qw = yaw_deg_to_quat(yaw_deg)
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y = qx, qy
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = qz, qw
        msg.pose.covariance[0] = cov
        msg.pose.covariance[7] = cov
        msg.pose.covariance[35] = cov * 0.4
        for _ in range(3):
            self.initpose_pub.publish(msg)
            rospy.sleep(0.5)

    def _recover(self):
        """2026-08-16 recovery: global costmap 定位错导致规划失败时,
        ①清除 costmap(错误障碍) ②重新定位。
        AMCL 模式: 发 /initialpose 重新撒粒子收敛;
        Cartographer 模式: 无粒子可撒, 且不订阅 /initialpose(发了没用),
        scan matching 逐帧自恢复, 跳过重定位只清 costmap。"""
        # 1. 清 costmap (消除"没有障碍的地方"的错误障碍标记)
        try:
            from std_srvs.srv import Empty
            proxy = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
            proxy()
            rospy.loginfo('[nav] recovery: 已清除 costmap')
        except Exception as e:
            rospy.logwarn('[nav] recovery: clear_costmaps 失败: %s', e)
        # 2. 重新定位 (仅 AMCL 模式; Cartographer 模式 scan matching 自恢复)
        if self._pose_src == 'amcl':
            if self._amcl_xy is not None and self._amcl_yaw is not None:
                self._publish_pose(self._amcl_xy[0], self._amcl_xy[1],
                                   math.degrees(self._amcl_yaw), cov=0.3)
                rospy.loginfo('[nav] recovery: 已用当前位姿(%.2f,%.2f,%.0f°)重新撒粒子',
                              self._amcl_xy[0], self._amcl_xy[1],
                              math.degrees(self._amcl_yaw))
            rospy.sleep(3.0)   # 等 AMCL 粒子收敛
        else:
            # Cartographer 模式: 逐帧 scan matching 自恢复, 只需清 costmap 后重试
            rospy.loginfo('[nav] recovery: Cartographer 模式, 跳过重定位 (scan matching 自恢复)')
            rospy.sleep(1.0)   # 短暂等 costmap 更新

    def _publish_goal_marker(self, x, y, yaw_deg):
        """发布当前目标航点的箭头 marker (rviz 可视化)。"""
        try:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = rospy.Time.now()
            m.ns = 'nav_goal'
            m.id = 0
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.15
            qx, qy, qz, qw = yaw_deg_to_quat(yaw_deg)
            m.pose.orientation.x, m.pose.orientation.y = qx, qy
            m.pose.orientation.z, m.pose.orientation.w = qz, qw
            m.scale.x = 0.35     # 箭头长度
            m.scale.y = 0.06     # 宽度
            m.scale.z = 0.06
            m.color.r = 0.0
            m.color.g = 1.0     # 绿色箭头
            m.color.b = 0.0
            m.color.a = 1.0
            self.goal_marker_pub.publish(m)
        except Exception as e:
            rospy.logwarn_throttle(5.0, '[nav] goal marker 发布失败: %s', e)

    # ---------------- 启动：map + 初始位姿 ----------------
    def _wait_map(self):
        got = []

        def cb(_m):
            got.append(1)

        sub = rospy.Subscriber(MAP_TOPIC, OccupancyGrid, cb)
        t0 = time.time()
        while not got and time.time() - t0 < self.map_wait_s and not rospy.is_shutdown():
            rospy.sleep(0.2)
        sub.unregister()
        if not got:
            rospy.logerr('[nav] %.0fs 内未收到 %s, AMCL 可能未收敛, 继续', self.map_wait_s, MAP_TOPIC)
        else:
            rospy.loginfo('[nav] 已收到 %s', MAP_TOPIC)

    def _publish_initial_pose(self):
        x = float(self.ip.get('x', 0.0))
        y = float(self.ip.get('y', 0.0))
        yaw = float(self.ip.get('yaw_deg', 0.0))
        # 初始位姿用较大方差 (车摆放有偏差时不锁死, 靠窄道板子特征自行收敛)
        self._publish_pose(x, y, yaw, cov=0.5)
        rospy.loginfo('[nav] 已发布 AMCL 初始位姿 (%.2f, %.2f, %.1f°)', x, y, yaw)

    # ---------------- move_base ----------------
    def _connect_move_base(self):
        import actionlib
        self.client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        rospy.loginfo('[nav] 等待 move_base action server ...')
        if not self.client.wait_for_server(rospy.Duration(30)):
            rospy.logfatal('[nav] move_base 不可用, 无法导航, 退出')
            rospy.signal_shutdown('move_base unavailable')
            raise RuntimeError('move_base unavailable')
        rospy.loginfo('[nav] move_base 已连接')

    def _make_goal(self, x, y, yaw_deg):
        g = MoveBaseGoal()
        g.target_pose.header.frame_id = 'map'
        g.target_pose.header.stamp = rospy.Time.now()
        g.target_pose.pose.position.x = x
        g.target_pose.pose.position.y = y
        g.target_pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_deg_to_quat(yaw_deg)
        g.target_pose.pose.orientation.x, g.target_pose.pose.orientation.y = qx, qy
        g.target_pose.pose.orientation.z, g.target_pose.pose.orientation.w = qz, qw
        return g

    def _nav_to(self, wp):
        """导航到航点; 返回 True 到达 / False 放弃。Ⅳ 走廊启用横向绕行兜底。
        2026-08-15: 无任务中间点(路径引导点)失败不重试、超时 20s 快速跳过;
        任务点照常 45s + 重试。"""
        name = wp['name']
        iv = bool(wp.get('iv', False))
        has_tasks = bool(wp.get('tasks'))
        self._set_state('NAV:%s' % name)
        # 2026-08-16: rviz 可视化当前目标航点 (绿色箭头, 位置+朝向)
        self._publish_goal_marker(wp['x'], wp['y'], wp['yaw_deg'])

        # 常规重试
        # 2026-08-15: iv 航点(四区走廊)导航超时缩短到 20s —— 被盘堵死时
        # 不必白等 2×45s 才走横向绕行兜底
        timeout = 20.0 if iv else self.nav_timeout_s
        retries = self.nav_retry if has_tasks else 0   # 中间点不重试
        for attempt in range(retries + 1):
            if self._budget.exhausted():
                return False
            # 中间点用短超时 (快速跳过, 不卡全场)
            use_timeout = min(timeout, 20.0) if not has_tasks else timeout
            self.client.send_goal(self._make_goal(wp['x'], wp['y'], wp['yaw_deg']))
            finished = self.client.wait_for_result(rospy.Duration(use_timeout))
            st = self.client.get_state()
            if finished and st == 3:
                self._last_reached = wp
                return True
            # 2026-08-16 兜底: move_base 判未达, 但车实际已在目标 0.2m 内
            # (AMCL 定位偏差导致 move_base 判 ABORTED/超时)。
            # ★ 只对无任务中间点兜底 (朝向不重要); 任务点朝向关键, 必须真正
            # 转到目标朝向 (否则云台按车头偏 60° 拍, 识别不到人偶)。
            if (not has_tasks and self._close_to(wp['x'], wp['y'], 0.2)):
                rospy.logwarn('[nav] %s move_base未达但车已在目标0.2m内, 视为到达',
                              wp['name'])
                self._last_reached = wp
                return True
            # 2026-08-15: 只在 PENDING(0)/ACTIVE(1) 时 cancel —— goal 已进入
            # 终态(ABORTED 等)再 cancel 会触发 "PREEMPTING when DONE" 状态机错误
            if st in (0, 1):
                self.client.cancel_goal()
            rospy.logwarn('[nav] 目标 (%.2f, %.2f) 未达 (state=%s)',
                          wp['x'], wp['y'], GOAL_STATUS_STR.get(st, st))
            rospy.sleep(1.0)

        # 2026-08-16 recovery: 重试都失败后, 清 costmap + 重新定位, 最后试一次
        # (global costmap 定位错导致 global planner 规划不了路线时, 清掉错误
        # 障碍 + 重新撒粒子收敛, 往往能恢复)
        if not self._budget.exhausted():
            self._recover()
            self.client.send_goal(self._make_goal(wp['x'], wp['y'], wp['yaw_deg']))
            finished = self.client.wait_for_result(rospy.Duration(timeout))
            st = self.client.get_state()
            if finished and st == 3:
                self._last_reached = wp
                return True
            if (not has_tasks and self._close_to(wp['x'], wp['y'], 0.2)):
                rospy.logwarn('[nav] %s recovery后车在目标0.2m内, 视为到达', wp['name'])
                self._last_reached = wp
                return True
            if st in (0, 1):
                self.client.cancel_goal()
            rospy.logwarn('[nav] %s recovery后仍未达 (state=%s)', wp['name'],
                          GOAL_STATUS_STR.get(st, st))

        # Ⅳ 走廊: move_base 被盘堵死时横向绕行兜底 (绝不碰盘, 绕不过就放弃 D 交通)
        if iv:
            if self._iv_retry(wp):
                self._last_reached = wp
                return True
            rospy.logwarn('[nav] %s: 走廊绕行多次失败, 放弃该区(不碰盘), 前往下一站', name)

        return False

    def _iv_retry(self, wp):
        """Ⅳ 走廊反坦克兜底: 等陈旧云消失 -> 横向临时目标 (垂直到达段方向 ±offset)
        -> 到横向目标后再去真实目标。全失败返回 False。"""
        rospy.sleep(self.iv_stall_wait_s)   # 等 observation_persistence 到期, 陈旧云自清
        if self._last_reached is None:
            return False
        lat_goals = lateral_offset_goals(self._last_reached, wp, self.iv_lateral_m)
        if not lat_goals:
            return False
        for i, (lx, ly) in enumerate(lat_goals):
            if self._budget.exhausted():
                return False
            self._set_state('NAV:%s-横向%d' % (wp['name'], i + 1))
            rospy.loginfo('[nav] %s 横向绕行 #%d -> (%.2f, %.2f)', wp['name'], i + 1, lx, ly)
            if self._nav_to_temp(lx, ly, wp['yaw_deg']):
                # 到横向点后再回真实目标
                return self._nav_to_temp(wp['x'], wp['y'], wp['yaw_deg'])
        return False

    def _nav_to_temp(self, x, y, yaw_deg):
        """临时目标导航 (不递归 _nav_to 以免 _last_reached 被覆盖)。"""
        self.client.send_goal(self._make_goal(x, y, yaw_deg))
        finished = self.client.wait_for_result(rospy.Duration(self.nav_timeout_s))
        st = self.client.get_state()
        if finished and st == 3:
            return True
        if st in (0, 1):   # 2026-08-15: 终态不 cancel (防 PREEMPTING when DONE)
            self.client.cancel_goal()
        return False

    def _align_yaw(self, target_yaw_deg):
        """任务点朝向对齐 (2026-08-16): move_base 到达后车头 yaw 若与目标偏差
        >阈值, 原地转向对齐 —— 云台按车头偏角识别, 朝向不对识别不到。"""
        if self._amcl_yaw is None:
            return
        target = math.radians(float(target_yaw_deg))
        diff = math.degrees(_angle_diff(self._amcl_yaw, target))
        if abs(diff) <= YAW_ALIGN_TOL_DEG:
            return
        rospy.loginfo('[nav] 任务点朝向偏差 %.1f° > %.1f°, 原地转向对齐 ...',
                      diff, YAW_ALIGN_TOL_DEG)
        self._set_state('ALIGN:朝向对齐%.1f°' % diff)
        if self._amcl_xy is None:
            return
        self.client.send_goal(self._make_goal(
            self._amcl_xy[0], self._amcl_xy[1], target_yaw_deg))
        self.client.wait_for_result(rospy.Duration(10.0))

    # ---------------- 任务触发 (方案D: 服务同步调用) ----------------
    def _call_service_timeout(self, proxy, zone, gimbal_bias, timeout_s):
        """服务调用总超时保护: 服务端内部有硬超时(人物~30s/雷区~5s),
        这里再包一层客户端超时(内部超时 + 5s 余量), 防服务端异常卡死
        把整个导航流程永久阻塞。返回响应对象或 None(超时)。"""
        result = {}

        def worker():
            try:
                result['resp'] = proxy(zone, gimbal_bias)
            except Exception as e:
                result['err'] = e

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        th.join(timeout_s + 5.0)
        if th.is_alive():
            rospy.logwarn('[nav] %s 服务调用超时(>%.0fs), 放弃等待', proxy.resolved_name,
                          timeout_s + 5.0)
            return None
        if 'err' in result:
            raise result['err']
        return result.get('resp')

    def _run_task(self, task, gimbal_bias=0.0):
        t = task['type']
        # 2026-08-14: 视觉避障开关任务 (Ⅳ区入口开启, 终点关闭)
        if t in ('avoid_on', 'avoid_off'):
            return self._run_avoid_toggle(t)
        # 2026-08-16: 反坦克一次性识别 (Ⅳ区 MineB 下一个点停稳后调一次)
        if t == 'antitank':
            return self._run_antitank()
        z = task['zone']
        label = '人物' if t == 'person' else '雷区'
        service = PERSON_SERVICE if t == 'person' else MINE_SERVICE
        timeout = self.person_timeout_s if t == 'person' else self.mine_timeout_s
        self._set_state('TASK:%s%s' % (label, z))
        # 2026-08-15 关键修复: python3 的 rospy.wait_for_service 返回 None(不可靠),
        # 即使服务存在也判"不可用"。改用直接 ServiceProxy 调用 + 异常重试。
        proxy = rospy.ServiceProxy(service, DetectTask)
        resp = None
        for attempt in range(4):
            try:
                resp = self._call_service_timeout(proxy, z, gimbal_bias, timeout)
                break
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logwarn('[nav] %s%s 服务调用失败(%d/4): %s, 5s后重试',
                              label, z, attempt + 1, e)
                rospy.sleep(5.0)
        if resp is None:
            rospy.logwarn('[nav] %s%s 服务不可用(重试4次), 报告补 0', label, z)
            return None
        if resp.success:
            rospy.loginfo('[nav] %s%s 报告: %s', label, z, resp.report)
            return resp.report
        rospy.logwarn('[nav] %s%s 服务返回失败: %s (报告补 0)', label, z, resp.message)
        return None

    def _run_avoid_toggle(self, t):
        """视觉避障开关 (2026-08-14): /visual_mine/set_enabled 服务。
        Ⅳ区入口开启, 终点关闭。失败不阻塞主流程。
        2026-08-15: 不用 wait_for_service(python3 下不可靠), 直接调用+重试。"""
        want = (t == 'avoid_on')
        self._set_state('TASK:避障%s' % ('开启' if want else '关闭'))
        proxy = rospy.ServiceProxy(AVOID_TOGGLE_SERVICE, SetBool)
        resp = None
        for attempt in range(3):
            try:
                resp = proxy(want)
                break
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logwarn('[nav] 避障开关调用失败(%d/3): %s', attempt + 1, e)
                rospy.sleep(3.0)
        if resp is None:
            rospy.logwarn('[nav] 避障开关服务不可用, 跳过')
            return None
        rospy.loginfo('[nav] 视觉避障已%s: %s', '开启' if want else '关闭',
                      resp.message if resp else '?')
        return resp.message

    def _run_antitank(self):
        """Ⅳ区反坦克一次性识别 (2026-08-16): 调 /antitank_check_service,
        服务端识别两个盘 -> map 坐标 -> 发布点云常驻 map 供 costmap 避障。"""
        self._set_state('TASK:反坦克识别')
        proxy = rospy.ServiceProxy(AT_SERVICE, DetectTask)
        resp = None
        for attempt in range(3):
            try:
                resp = self._call_service_timeout(proxy, 'IV', 0.0, 10.0)
                break
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logwarn('[nav] 反坦克服务调用失败(%d/3): %s, 3s后重试',
                              attempt + 1, e)
                rospy.sleep(3.0)
        if resp is None:
            rospy.logwarn('[nav] 反坦克服务不可用, 跳过 (点云不发布)')
            return None
        if resp.success:
            rospy.loginfo('[nav] 反坦克识别完成: %s', resp.report)
            return resp.report
        rospy.logwarn('[nav] 反坦克识别失败: %s', resp.message)
        return None

    # ---------------- 主流程 ----------------
    def run(self):
        self._connect_move_base()
        self._wait_map()
        self._publish_initial_pose()
        # 2026-08-15: 初始位姿后留 3s 让 AMCL 粒子收敛再出发
        # (车实际摆放与初始位姿有偏差时, 直接出发第一段容易漂)
        rospy.loginfo('[nav] 等 AMCL 收敛 3s ...')
        rospy.sleep(3.0)

        total = len(self.course)
        for idx, wp in enumerate(self.course):
            if rospy.is_shutdown():
                break
            if self._budget.exhausted():
                rospy.logwarn('[nav] 全局预算 %.0fs 耗尽, 跳过剩余 %d 站, 进入收尾',
                              self.global_budget_s, total - idx)
                break
            # 发布当前站名 (供视觉避障节点做四区反坦克盘推理门控)
            self.zone_pub.publish(String(wp['name']))
            ok = self._nav_to(wp)
            if not ok:
                rospy.logwarn('[nav] 未到达 %s, 该站任务跳过 (报告自动补 0)', wp['name'])
                continue
            # 2026-08-16: 任务点朝向对齐 (识别/打靶依赖车头朝向, 朝向不对识别不到)
            if wp.get('tasks'):
                self._align_yaw(wp['yaw_deg'])
            for task in wp.get('tasks', []):
                if self._budget.exhausted():
                    rospy.logwarn('[nav] 预算耗尽, 跳过剩余任务')
                    break
                # 传递该站的云台预偏角 (person 任务用; mine 忽略)
                bias = float(wp.get('gimbal_bias', 0.0))
                self._run_task(task, gimbal_bias=bias)
            if wp.get('iv'):
                rospy.loginfo('[nav] Ⅳ 走廊避障状态: %s', self._avoid_status)

        self._final_report()
        self._set_state('DONE')
        rospy.loginfo('[nav] ======== 全流程结束, 总耗时 %.1fs ========',
                      self._budget.elapsed())

    # ---------------- 收尾 ----------------
    def _final_report(self):
        self._set_state('REPORT:合并报告')
        try:
            import rospkg
            pkg = rospkg.RosPack().get_path('raicom2026')
        except Exception:
            pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_dir = os.path.join(pkg, 'data', 'reports')
        text = assemble_report(
            os.path.join(report_dir, 'report_person.txt'),
            os.path.join(report_dir, 'report_mine.txt'),
            os.path.join(report_dir, 'report.txt'))
        lines = text.strip().split('\n')
        ok = len(lines) == 6
        rospy.loginfo('[nav] report.txt 已生成 (6 行=%s):\n%s', ok, text)
        if not ok:
            rospy.logerr('[nav] report.txt 行数 != 6, 需人工检查!')

    def print_course(self):
        """--print-course: 打印解析后的航点课程 (不下车/不连 move_base)。"""
        print('== 航点课程 (%d 站) ==' % len(self.course))
        for wp in self.course:
            tasks = ', '.join('%s:%s' % (t['type'], t.get('zone', '')) for t in wp.get('tasks', [])) or '(无任务)'
            iv = ' [四区·绕行兜底]' if wp.get('iv') else ''
            zone = wp.get('zone', '')
            bias = wp.get('gimbal_bias')
            bias_s = (' 云台偏角%+.0f°' % bias) if bias else ''
            print('  %-6s (%6.2f, %6.2f) yaw=%5.0f°  [%s]  %s%s%s'
                  % (wp['name'], wp['x'], wp['y'], wp['yaw_deg'], zone, tasks, iv, bias_s))
        print('== 初始位姿: x=%.2f y=%.2f yaw=%.0f° =='
              % (self.ip.get('x', 0), self.ip.get('y', 0), self.ip.get('yaw_deg', 0)))
        print('== 预算 %.0fs | 每段 %.0fs 重试 %d 次 | 人物超时 %.0fs 雷区超时 %.0fs =='
              % (self.global_budget_s, self.nav_timeout_s, self.nav_retry,
                 self.person_timeout_s, self.mine_timeout_s))


def print_course_from_file(path):
    """--print-course: 不连 master, 直接解析 yaml 校验并打印航点课程。
    用于实机改配置后的快速自查 (等价于 roslaunch 里 load 后节点启动时校验)。"""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    course = cfg.get('course', [])
    validate_course(course)
    ip = cfg.get('amcl_initial_pose', {})
    print('== 航点课程 (%d 站) ==' % len(course))
    for wp in course:
        tasks = ', '.join('%s:%s' % (t['type'], t.get('zone', '')) for t in wp.get('tasks', [])) or '(无任务)'
        iv = ' [四区·绕行兜底]' if wp.get('iv') else ''
        zone = wp.get('zone', '')
        bias = wp.get('gimbal_bias')
        bias_s = (' 云台偏角%+.0f°' % bias) if bias else ''
        print('  %-6s (%6.2f, %6.2f) yaw=%5.0f°  [%s]  %s%s%s'
              % (wp['name'], wp['x'], wp['y'], wp['yaw_deg'], zone, tasks, iv, bias_s))
    print('== 初始位姿: x=%.2f y=%.2f yaw=%.0f° =='
          % (ip.get('x', 0), ip.get('y', 0), ip.get('yaw_deg', 0)))
    print('== 人物超时 %.0fs | 雷区超时 %.0fs | 每段 %.0fs 重试 %d 次 | 全局预算 %.0fs =='
          % (cfg.get('person_timeout_s', 25), cfg.get('mine_timeout_s', 12),
             cfg.get('nav_timeout_s', 45), cfg.get('nav_retry', 1),
             cfg.get('global_budget_s', 280)))
    print('== 配置校验通过 ==')


def main():
    parser = argparse.ArgumentParser(description='RAICOM2026 自主导航编排节点')
    parser.add_argument('--print-course', action='store_true',
                        help='只解析 param/nav_competition.yaml 并打印航点课程后退出 (不连 master/不下车)')
    parser.add_argument('--config', default=None,
                        help='--print-course 用的 yaml 路径 (默认 param/nav_competition.yaml)')
    args = parser.parse_args(rospy.myargv(sys.argv)[1:])

    if args.print_course:
        cfg_path = args.config or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'param', 'nav_competition.yaml')
        print_course_from_file(cfg_path)
        return

    if not ensure_master():
        rospy.logwarn('[nav] 无法连接/拉起 master')

    nav = CompetitionNav()   # 内部已 init_node

    # rospy.init_node 会覆盖 SIGINT/SIGTERM 信号处理, 必须在 init 之后重注册
    # (记忆: rospy/core.py:623)。收到信号: 取消当前 move_base 目标 -> 请求关闭,
    # 由 run() 各循环点的 is_shutdown() 检查退出, 车立即停。
    def _sig(sig, _f):
        rospy.logwarn('[nav] 收到信号 %s, 取消当前导航目标并退出', sig)
        try:
            if hasattr(nav, 'client'):
                nav.client.cancel_goal()
        except Exception:
            pass
        rospy.signal_shutdown('signal %s' % sig)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        nav.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr('[nav] 主流程异常: %s', e)
        raise


if __name__ == '__main__':
    main()
