#!/bin/bash
# ============================================================
# stop_person.sh —— 清理所有节点 (2026-08-15 v2)
# ============================================================
# 覆盖: 测试模式(start_person) + 比赛 prep 的全部节点。
# 用法: bash /home/mowen/stop_person.sh
# ============================================================
echo "清理所有节点..."

# 检测节点
pkill -9 -f 'webcam_person_gun.py' 2>/dev/null
pkill -9 -f 'orbbec_mine_antitank.py' 2>/dev/null
pkill -9 -f 'visual_mine_avoidance.py' 2>/dev/null
pkill -9 -f 'cmd_vel_merge.py' 2>/dev/null
pkill -9 -f 'gimbal_server.py' 2>/dev/null
pkill -9 -f 'camera_bridge.py' 2>/dev/null
pkill -9 -f 'usb_cam' 2>/dev/null
pkill -9 -f 'nav_competition.py' 2>/dev/null
pkill -9 -f 'test_tf_pub.py' 2>/dev/null

# 底盘/导航节点 (prep)
pkill -9 -f 'move_base' 2>/dev/null
pkill -9 -f 'amcl' 2>/dev/null
pkill -9 -f 'ekf_localization' 2>/dev/null
pkill -9 -f 'lslidar' 2>/dev/null
pkill -9 -f 'base_controller' 2>/dev/null
pkill -9 -f 'imu_node' 2>/dev/null
pkill -9 -f 'odometry_publisher' 2>/dev/null
pkill -9 -f 'map_server' 2>/dev/null
pkill -9 -f 'robot_state_publisher' 2>/dev/null
pkill -9 -f 'static_transform_publisher' 2>/dev/null   # bringup 静态TF, 易残留冲突

# 相机驱动
pkill -9 -f 'dabai_dcw2' 2>/dev/null
pkill -9 -f 'orbbec_camera' 2>/dev/null

# roslaunch (最后杀, 会带死所有残留子节点)
pkill -9 -f 'roslaunch' 2>/dev/null
pkill -9 -f 'rviz' 2>/dev/null
# 连 rosmaster 一起杀 (2026-08-16: SIGKILL 的节点在 master 里残留僵尸注册,
# 不杀 master 新 prep 起同名节点会全部冲突)
pkill -9 -f 'rosmaster' 2>/dev/null
pkill -9 -f 'roscore' 2>/dev/null

sleep 2

# 检查残留
LEFT=$(ps aux | grep -E '[w]ebcam_person_gun|[o]rbbec_mine|[v]isual_mine|[m]ove_base|[a]mcl|[l]slidar|[g]imbal_server|[e]kf_localization|[r]osmaster' | wc -l)
if [ "$LEFT" -eq 0 ]; then
    echo "✅ 已清理干净 (含 roscore)"
else
    echo "仍有 $LEFT 个残留进程:"
    ps aux | grep -E '[w]ebcam_person_gun|[o]rbbec_mine|[v]isual_mine|[m]ove_base|[a]mcl|[l]slidar|[g]imbal_server|[e]kf_localization|[r]osmaster'
fi
