#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# carto_relocalize.py —— rviz 2D Pose Estimate 手动重新定位 Cartographer
# ------------------------------------------------------------
# 背景: Cartographer 1.0 不订阅 /initialpose (那是 AMCL 的话题),
# 所以 rviz 的 2D Pose Estimate 对 Cartographer 无效。
# 本节点订阅 /initialpose, 收到后:
#   1) finish 当前 ACTIVE 轨迹
#   2) 调 /start_trajectory 服务, 用 2D Pose Estimate 的位姿作初始位姿
#      重新 start 定位轨迹 -> Cartographer 从新位姿重新收敛定位。
#
# 用法: prep 起好后自动由 launch 启动; 或手动:
#   rosrun raicom2026 carto_relocalize.py
# 使用: rviz 工具栏点 "2D Pose Estimate", 在地图上点车实际位置 + 拖朝向箭头。
# ============================================================
import os
import rospkg
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from cartographer_ros_msgs.srv import (
    StartTrajectory, StartTrajectoryRequest,
    FinishTrajectory, GetTrajectoryStates,
)

CFG_DIR = os.path.join(rospkg.RosPack().get_path('raicom2026'), 'config', 'cartographer')


def main():
    rospy.init_node('carto_relocalize', anonymous=False)
    rospy.sleep(2.0)   # 等 cartographer_node 把服务起好

    def cb(msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        rospy.loginfo('[reloc] 收到 2D Pose Estimate (%.2f, %.2f)', x, y)
        try:
            # 1. finish 所有 ACTIVE 轨迹 (避免累积)
            states_srv = rospy.ServiceProxy('/get_trajectory_states',
                                            GetTrajectoryStates)
            states = states_srv()
            ts = states.trajectory_states
            for tid, tstate in zip(ts.trajectory_id, ts.trajectory_state):
                if tstate == 0:   # ACTIVE
                    finish_srv = rospy.ServiceProxy('/finish_trajectory',
                                                    FinishTrajectory)
                    finish_srv(tid)
                    rospy.loginfo('[reloc] finish 旧轨迹 %d', tid)
            # 2. start 新轨迹 (initial_pose = 2D Pose Estimate 的 map 系位姿)
            srv = rospy.ServiceProxy('/start_trajectory', StartTrajectory)
            req = StartTrajectoryRequest()
            req.configuration_directory = CFG_DIR
            req.configuration_basename = 'localization.lua'
            req.use_initial_pose = True
            req.initial_pose = msg.pose.pose
            req.relative_to_trajectory_id = 0   # 相对 load_state 的建图轨迹(0)
            resp = srv(req)
            rospy.loginfo('[reloc] 重新定位: %s (trajectory_id=%d)',
                          resp.status.message, resp.trajectory_id)
        except Exception as e:
            rospy.logwarn('[reloc] 重新定位失败: %s', e)

    rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, cb)
    rospy.loginfo('[reloc] 就绪, 等 rviz 2D Pose Estimate (/initialpose)')
    rospy.spin()


if __name__ == '__main__':
    main()
