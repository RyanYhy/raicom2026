#!/bin/bash
# ============================================================
# clean_ros.sh —— 测试前清理 ROS 环境 (杀干净所有残留进程)
# ------------------------------------------------------------
# 用法 (每次测试/重启 prep 前跑):
#   bash /home/mowen/clean_ros.sh
# 作用: 杀掉所有 ROS 相关进程 + 等端口释放 + 确认干净,
#       避免"残留进程累积 -> 同名冲突 -> bringup 节点起不来"。
# ============================================================

echo "========== 清理 ROS 环境 =========="

# 所有可能残留的 ROS 进程 (进程名/命令行关键词)
PROC_PATTERN='cartographer_node|rosmaster|roscore|rosout|roslaunch|map_server|move_base|lslidar|ekf_localization|robot_localization|orbbec|dabai|base_controller|odometry_publisher|base_node|robot_state_publisher|static_transform_publisher|imu_node|wit_multi|visual_mine|webcam_person|orbbec_mine|camera_bridge|gimbal_server|cmd_vel_merge|carto_init|odom_cleaner|rviz|test_tf_pub|nav_competition|start_match_prep|start_match_go'

# 停稳小车：先发 0 速度，再杀进程 (此时 roscore 若还活着才能发出)
echo "发布 cmd_vel = 0 ..."
timeout 3 rostopic pub -1 /cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
  2>/dev/null || true


echo "杀掉 ROS 进程 ..."
KILLED=0
for p in $(ps aux | grep -E "$PROC_PATTERN" | grep -v grep | awk '{print $2}'); do
    kill -9 $p 2>/dev/null
    KILLED=$((KILLED+1))
done
echo "  已杀 $KILLED 个进程"

# 等进程退出 + 端口释放
sleep 6

# 确认干净
REMAIN=$(ps aux | grep -E 'ros|carto|lslidar|orbbec' | grep -v grep | wc -l)
PORT=$(ss -tlnp 2>/dev/null | grep 11311 | wc -l)

echo ""
if [ "$REMAIN" -eq 0 ] && [ "$PORT" -eq 0 ]; then
    echo "✅ 环境干净 (0 残留进程, 端口 11311 空闲)"
else
    echo "⚠️ 残留进程 $REMAIN 个, 端口 11311 占用 $PORT 处"
    echo "  残留进程:"
    ps aux | grep -E 'ros|carto|lslidar|orbbec' | grep -v grep
fi

echo "========== 清理完成 =========="
