#!/bin/bash
# ============================================================
# start_match_go.sh —— 比赛出发段 (1raicom_ws)
# ============================================================
# 裁判计时开始瞬间运行: 清空旧报告 + 只启动导航编排节点, 秒级出发。
# 前置: 赛前已跑 start_match_prep.sh 且看到所有服务就绪。
# ★ Ctrl+C = 自动停车并清理导航节点
# 用法: bash /home/mowen/1raicom_ws/scripts/start_match_go.sh
# ============================================================
source /home/mowen/1raicom_ws/scripts/setup_env.sh

REPORT_DIR=/home/mowen/1raicom_ws/src/raicom2026/data/reports
WS_SCRIPTS=/home/mowen/1raicom_ws/scripts

# ---- 退出时自动清理导航节点 ----
cleanup() {
  echo ""
  echo "== 退出中: 清理导航节点 (外设/模型节点保留, prep 终端可继续用) =="
  pkill -9 -f 'nav_competition.py' 2>/dev/null
  pkill -9 -f 'match_go[.]launch' 2>/dev/null
  echo "== 导航节点已清理 =="
}
trap cleanup EXIT INT TERM

# 清空旧报告 (报告文件是增量写的, 不清会把上次测试数据算进这次)
rm -f "${REPORT_DIR}/report_person.txt"
rm -f "${REPORT_DIR}/report_mine.txt"
rm -f "${REPORT_DIR}/report.txt"
echo "已清空旧报告文件"

echo "等待检测服务就绪 (person/mine 模型加载中)..."
READY=0
for i in $(seq 1 45); do
    if rosservice list 2>/dev/null | grep -q person_detect_service; then
        READY=1
        echo "✅ 检测服务已就绪 (等了 $((i*2)) 秒)"
        break
    fi
    sleep 2
done
if [ "$READY" -eq 0 ]; then
    echo "⚠️ 90 秒内服务仍未就绪, 检查 prep 是否在运行 (bash ${WS_SCRIPTS}/start_match_prep.sh)"
    echo "   继续出发 (服务会在到站时报'不可用'补 0)"
fi

# 出发: 只起导航编排节点
roslaunch raicom2026 match_go.launch
