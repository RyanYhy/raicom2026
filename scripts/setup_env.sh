#!/bin/bash
# 1raicom_ws 统一环境入口 —— 所有终端 / prep / go 都应 source 此文件
source /opt/ros/melodic/setup.bash
source /home/mowen/cartographer_ws/install_isolated/setup.bash --extend
source /home/mowen/1raicom_ws/devel/setup.bash --extend
source /home/mowen/cv_bridge_ws/devel/setup.bash --extend
export ROS_MASTER_URI=http://localhost:11311

# cv_bridge 的 setup 链会带入 1competition_ws 等旧 ws，且排在 1raicom_ws 前面；
# 强制让 1raicom_ws 优先（同名包 rospack 取第一个匹配）
export ROS_PACKAGE_PATH=/home/mowen/1raicom_ws/src:${ROS_PACKAGE_PATH}
export CMAKE_PREFIX_PATH=/home/mowen/1raicom_ws/devel:${CMAKE_PREFIX_PATH}
