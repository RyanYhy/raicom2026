#!/bin/bash
# ============================================================
# save_pbstream.sh —— 建图完成后保存 Cartographer pbstream 地图
# ------------------------------------------------------------
# 用法 (建图完成、车停稳后, 单独开终端执行):
#   bash /home/mowen/1raicom_ws/scripts/save_pbstream.sh
#   bash .../save_pbstream.sh mymap        # 自定义地图名
# ============================================================
source /opt/ros/melodic/setup.bash
source /home/mowen/1raicom_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

MAP_NAME="${1:-competition_map}"
OUT_DIR="/home/mowen/1raicom_ws/src/raicom2026/data/maps"
OUT_FILE="$OUT_DIR/${MAP_NAME}.pbstream"

echo "=============================================="
echo "  保存 Cartographer pbstream 地图"
echo "  输出: $OUT_FILE"
echo "=============================================="

echo ""
echo "== 1. 结束当前轨迹 (trajectory 0) =="
rosservice call /finish_trajectory 0

sleep 2

echo ""
echo "== 2. 写入 pbstream =="
rosservice call /write_state "{filename: '$OUT_FILE', include_unfinished_submaps: false}"

sleep 2

echo ""
echo "== 3. 验证文件 =="
if [ -f "$OUT_FILE" ]; then
    ls -la "$OUT_FILE"
    echo ""
    echo "✅ pbstream 已保存: $OUT_FILE"
    echo "   接下来用这个文件做 Cartographer 纯定位"
else
    echo "❌ 保存失败! 检查 cartographer_node 是否在跑"
fi
