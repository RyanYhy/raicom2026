#!/bin/bash
# ============================================================
# check_devices.sh —— 外设插线检查 (2026-08-15)
# ============================================================
# 终端直接跑: bash /home/mowen/check_devices.sh
# 检查全部外设是否插好, ✅/❌ 一目了然。
# ============================================================

echo ""
echo "============= 外设插线检查 ============="
echo ""

# ---- 1. 串口设备 (符号链接) ----
check_serial() {
    local name="$1"
    local dev="$2"
    if [ -e "$dev" ]; then
        echo "  ✅ $name: $dev -> $(readlink -f "$dev" 2>/dev/null || echo "$dev")"
    else
        echo "  ❌ $name: $dev 缺失! (USB线未插好/掉线)"
    fi
}
echo "[1] 串口设备:"
check_serial "底盘"        /dev/carserial
check_serial "IMU"         /dev/imu
check_serial "云台/水弹枪" /dev/gun
check_serial "雷达"        /dev/lslidar

# ---- 2. USB 摄像头 (video 设备, 只排除 Orbbec; "Integrated Webcam" 就是 USB 摄像头) ----
echo ""
echo "[2] USB 摄像头:"
found=0
for v in /dev/video*; do
    [ -e "$v" ] || continue
    name=$(cat /sys/class/video4linux/$(basename "$v")/name 2>/dev/null)
    echo "      $v: $name"
    case "$name" in
        *Obsensor*) ;;
        *Orbbec*) ;;
        "") ;;
        *) found=1 ;;
    esac
done
if [ "$found" -eq 1 ]; then
    echo "  ✅ 找到摄像头 (注意: 实车 USB 摄像头产品名就叫 'Integrated Webcam', 属正常)"
else
    echo "  ❌ 未找到摄像头!"
fi

# ---- 3. Orbbec 深度相机 ----
echo ""
echo "[3] Orbbec 深度相机:"
if lsusb 2>/dev/null | grep -qi "2bc5"; then
    echo "  ✅ Orbbec (2bc5) 在 USB 总线"
else
    echo "  ❌ Orbbec 不在 USB 总线!"
fi

# ---- 4. ch341 串口芯片数量 (应 3 个: 底盘+IMU+云台) ----
echo ""
echo "[4] ch341 串口芯片:"
n=$(lsusb 2>/dev/null | grep -c "1a86:7523")
echo "      数量: $n (期望 3 = 底盘+IMU+云台)"
if [ "$n" -eq 3 ]; then
    echo "  ✅ 3 个齐全"
elif [ "$n" -eq 2 ]; then
    echo "  ❌ 少 1 个 (通常缺云台/水弹枪那个, 拔插它的 USB 线)"
else
    echo "  ❌ 异常: 只有 $n 个"
fi

# ---- 汇总 ----
echo ""
echo "============= 检查完成 ============="
echo ""
echo "提示: 所有 ✅ 才算齐。有 ❌ 的话拔插对应 USB 线后重跑本脚本。"
echo ""
