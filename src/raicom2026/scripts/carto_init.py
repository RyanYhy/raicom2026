#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# carto_init.py —— Cartographer 纯定位初始位姿注入
# ------------------------------------------------------------
# 背景: cartographer_node 启动时 start_trajectory_with_default_topics=false,
#       不自动 start 定位 trajectory。本脚本调 /start_trajectory 服务,
#       带 use_initial_pose=true + 起点位姿, 让定位从起点收敛 (而非
#       global localization 全局搜索, 后者在对称场地会定位错 ~85cm)。
#
# 用法 (prep 启动后):
#   rosrun raicom2026 carto_init.py
#   rosrun raicom2026 carto_init.py _x:=0.1 _y:=0.0 _yaw_deg:=0.0
# ============================================================
import sys
import math
import time

import os
import rospkg
import rospy
from cartographer_ros_msgs.srv import StartTrajectory, StartTrajectoryRequest
from geometry_msgs.msg import Pose


def main():
    rospy.init_node('carto_init', anonymous=True)
    x = rospy.get_param('~x', 0.0)
    y = rospy.get_param('~y', 0.0)
    yaw_deg = rospy.get_param('~yaw_deg', 0.0)
    yaw = math.radians(yaw_deg)

    # 2026-08-16: 不用 wait_for_service (python3 下不可靠, 立即返回导致节点退出),
    # 直接 ServiceProxy + 循环重试, 等 cartographer_node 把服务起好。
    srv = rospy.ServiceProxy('/start_trajectory', StartTrajectory)
    req = StartTrajectoryRequest()
    req.configuration_directory = os.path.join(rospkg.RosPack().get_path('raicom2026'), 'config', 'cartographer')
    req.configuration_basename = 'localization.lua'
    req.use_initial_pose = True
    req.initial_pose = Pose()
    req.initial_pose.position.x = x
    req.initial_pose.position.y = y
    req.initial_pose.position.z = 0.0
    req.initial_pose.orientation.x = 0.0
    req.initial_pose.orientation.y = 0.0
    req.initial_pose.orientation.z = math.sin(yaw / 2.0)
    req.initial_pose.orientation.w = math.cos(yaw / 2.0)
    # 相对 load_state 加载的 pbstream 轨迹 (ID 0, 建图轨迹)
    req.relative_to_trajectory_id = 0

    for attempt in range(15):
        try:
            resp = srv(req)
            rospy.loginfo('[carto_init] start_trajectory 成功: %s (trajectory_id=%d)',
                          resp.status.message, resp.trajectory_id)
            print('=' * 52)
            print('  Cartographer 初始位姿已注入: (%.2f, %.2f, %.1f°)' % (x, y, yaw_deg))
            print('  status: %s' % resp.status.message)
            print('=' * 52)
            return
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn('[carto_init] start_trajectory 调用失败(%d/15): %s', attempt + 1, e)
            time.sleep(2.0)
    rospy.logerr('[carto_init] 15 次调用失败, 初始位姿未注入')


if __name__ == '__main__':
    main()
