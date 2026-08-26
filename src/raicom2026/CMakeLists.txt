cmake_minimum_required(VERSION 3.0.2)
project(raicom2026)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  actionlib_msgs
  message_generation
  move_base_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf2
  tf2_ros
  nav_msgs
)

catkin_python_setup()

################################################
## 服务消息 (本包专用 /gun 服务 等)
################################################
add_service_files(
  FILES
  GunAim.srv
  GunFire.srv
  DetectTask.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs actionlib_msgs message_runtime move_base_msgs rospy sensor_msgs std_msgs std_srvs tf tf2 tf2_ros nav_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

################################################
## Python 脚本安装
## 新包全部节点脚本放在 scripts/，安装后可用 rosrun/roslaunch 调用
################################################
catkin_install_python(PROGRAMS
  scripts/gimbal_server.py
  scripts/calibrate.py
  scripts/odom_cleaner.py
  scripts/webcam_person_gun.py
  scripts/orbbec_mine_antitank.py
  scripts/visual_mine_avoidance.py
  scripts/nav_competition.py
  scripts/assemble_report.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 说明：
## - 本包只放"新写的"代码，参考 1205/robot_bringup 里的实现时只读，禁止修改原包。
## - scripts/ 下脚本随开发进度逐个加入上方 catkin_install_python(PROGRAMS ...) 列表。
## - 详见 docs/比赛方案与技术文档.md
