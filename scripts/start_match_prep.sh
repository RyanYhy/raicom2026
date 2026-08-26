#!/bin/bash
# ============================================================
# start_match_prep.sh —— 比赛赛前准备段 (1raicom_ws)
# ============================================================
# 后台启动全部外设/模型/Cartographer/move_base, 前台等待就绪后提示。
# ★ prep 在后台常驻, 本脚本等就绪后自动退出, 不占终端!
# ★ 清理节点: bash /home/mowen/1raicom_ws/scripts/stop_person.sh
# 用法: bash /home/mowen/1raicom_ws/scripts/start_match_prep.sh
# ============================================================
source /home/mowen/1raicom_ws/scripts/setup_env.sh

WS_SCRIPTS=/home/mowen/1raicom_ws/scripts

# ---- 启动前抢占式清理 (防同名冲突) ----
echo "清理残留节点..."
pkill -9 -f 'nav_competition.py' 2>/dev/null
pkill -9 -f 'webcam_person_gun.py' 2>/dev/null
pkill -9 -f 'orbbec_mine_antitank.py' 2>/dev/null
pkill -9 -f 'visual_mine_avoidance.py' 2>/dev/null
pkill -9 -f 'cmd_vel_merge.py' 2>/dev/null
pkill -9 -f 'gimbal_server.py' 2>/dev/null
pkill -9 -f 'camera_bridge.py' 2>/dev/null
pkill -9 -f 'usb_cam' 2>/dev/null
pkill -9 -f 'dabai_dcw2' 2>/dev/null
pkill -9 -f 'orbbec_camera' 2>/dev/null
pkill -9 -f 'test_tf_pub.py' 2>/dev/null
pkill -9 -f 'move_base' 2>/dev/null
pkill -9 -f 'amcl' 2>/dev/null
pkill -9 -f 'ekf_localization' 2>/dev/null
pkill -9 -f 'lslidar' 2>/dev/null
pkill -9 -f 'base_controller' 2>/dev/null
pkill -9 -f 'imu_node' 2>/dev/null
pkill -9 -f 'odometry_publisher' 2>/dev/null
pkill -9 -f 'map_server' 2>/dev/null
pkill -9 -f 'robot_state_publisher' 2>/dev/null
pkill -9 -f 'static_transform_publisher' 2>/dev/null
pkill -9 -f 'roslaunch' 2>/dev/null
pkill -9 -f 'rviz' 2>/dev/null
pkill -9 -f 'rosmaster' 2>/dev/null
pkill -9 -f 'roscore' 2>/dev/null
sleep 2
echo "开始启动..."

# ---- 后台启动 prep (不占终端) ----
setsid nohup roslaunch raicom2026 nav_competition_2026.launch start_nav:=false loc_mode:=cartographer \
  > /tmp/prep.log 2>&1 < /dev/null &
echo "prep 已在后台启动, 等待服务就绪..."

# ---- 前台等待就绪 (最多 120 秒) ----
READY=0
for i in $(seq 1 60); do
    if rosservice list 2>/dev/null | grep -q person_detect_service; then
        READY=1
        echo ""
        echo "=============================================="
        echo "  ✅ 服务已就绪 (等了 $((i*2)) 秒)"
        echo "  prep 正在后台运行, 现在可以跑 go 了:"
        echo "    bash ${WS_SCRIPTS}/start_match_go.sh"
        echo "=============================================="
        echo ""
        break
    fi
    sleep 2
done

if [ "$READY" -eq 0 ]; then
    echo "⚠️ 120 秒内服务未就绪, 检查 /tmp/prep.log 日志:"
    tail -20 /tmp/prep.log
fi
