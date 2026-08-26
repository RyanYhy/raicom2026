# RAICOM 2026 智能侦察 — ROS 工作空间

全国大学生机器人大赛（RAICOM）「智能侦察」赛项的整车软件仓库。基于 **ROS Melodic**，实现底盘驱动、激光/IMU 融合、Cartographer 定位、视觉侦察（人物/地雷/反坦克）与自动导航打靶。

## 硬件

| 设备 | 说明 | udev 节点 |
|------|------|-----------|
| 底盘 | 串口差速 | `/dev/carserial` |
| IMU | Wit 协议 | `/dev/imu` |
| 激光雷达 | 镭神 N10_P | `/dev/lslidar` |
| 云台/水弹枪 | 串口 | `/dev/gun` |
| USB 摄像头 | 人物识别/打靶 | `/dev/video*` |
| Orbbec DaBai DCW2 | 深度相机（地雷/反坦克/避障） | USB `2bc5` |

udev 规则见 `config/udev/`。换车或换 USB 口后需重新核对规则。

## 目录结构

```
1raicom_ws/
├── src/
│   ├── raicom2026/     # 比赛应用：导航编排、视觉检测、launch
│   ├── robot_bringup/        # 底盘/雷达/IMU 启动 launch
│   ├── robot_navigation/     # move_base、EKF 参数
│   ├── robot_description/    # URDF 与 mesh
│   ├── chassis_driver/       # 底盘串口驱动
│   ├── imu_driver/           # IMU 驱动
│   ├── car_bringup/          # 里程计转换
│   ├── orbbec_camera/        # Orbbec 深度相机驱动
│   └── leishen/              # 镭神雷达驱动 (lslidar*)
├── scripts/                  # 比赛与运维脚本 (prep / go / clean / setup_env)
├── test/                     # 开发标定脚本与调试 launch（比赛不加载）
├── docs/                     # 文档
└── config/udev/              # 设备 udev 规则
```

## 环境要求

- Ubuntu 18.04 + ROS Melodic
- 额外 overlay（需单独编译，不在本 ws 内）：
  - `cartographer_ws` — Cartographer 定位
  - `cv_bridge_ws` — Python3 版 cv_bridge（YOLO 节点用）
- Conda 环境 `yolov8`（ONNX 推理）

## 快速开始

```bash
# 1. 编译
cd ~/1raicom_ws
source /opt/ros/melodic/setup.bash
catkin_make
# 若 lslidar 首次并行编译失败，再执行一次 catkin_make

# 2. 加载环境（每个新终端）
source ~/1raicom_ws/scripts/setup_env.sh

# 3. 检查外设
bash ~/1raicom_ws/scripts/check_devices.sh

# 4. 比赛两段式启动
bash ~/1raicom_ws/scripts/start_match_prep.sh   # 赛前：外设+模型+定位+move_base
bash ~/1raicom_ws/scripts/start_match_go.sh       # 出发：仅导航编排
```

清理 ROS 进程：`bash ~/1raicom_ws/scripts/clean_ros.sh`

## 比赛流程

1. **prep** — 后台启动 `nav_competition_2026.launch`（`start_nav:=false`），加载全部传感器、YOLO 模型、Cartographer 定位与 move_base，前台等待检测服务就绪后退出。
2. **go** — 清空报告、启动 `match_go.launch`（`nav_competition.py`），按航点巡航并触发人物/地雷/反坦克任务。
3. 报告输出：`src/raicom2026/data/reports/`

定位模式在 launch 中通过 `loc_mode:=cartographer`（默认）或 `amcl` 切换；地图与 pbstream 在 `raicom2026/data/maps/`。

## ROS 包说明

| 包 | 职责 |
|----|------|
| `raicom2026` | 比赛逻辑：导航编排、视觉检测、launch 入口 |
| `robot_bringup` | `bringup_robot.launch`：雷达 + IMU + 底盘 + EKF + URDF |
| `robot_navigation` | costmap / move_base / DWA / EKF 参数 yaml |
| `robot_description` | 小车 URDF (`mowen.urdf.xacro`) |
| `chassis_driver` | `base_controller.py` |
| `imu_driver` | `wit_multi_ros.py` |
| `orbbec_camera` | Orbbec 深度相机 ROS 驱动 |
| `car_bringup` | 里程计 `base_node` |
| `lslidar_driver` | 镭神雷达驱动 |

## 开发与标定

标定、采数据、标航点等工具在 `test/scripts/`；调试 launch 在 `test/launch/`。使用前同样 `source scripts/setup_env.sh`。

详细操作见 `docs/`（建图、识别打靶、外设注意事项等）。

## 模型与地图

- ONNX 模型：`src/raicom2026/data/models/`（`person.onnx`、`mine.onnx`、`antitank.onnx`）
- 比赛地图：`src/raicom2026/data/maps/map2.yaml` + `map2.pbstream`
- YOLO 训练集配置：`data/data_*.yaml`（训练数据另行归档，不在仓库内）

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/比赛方案与技术文档.md` | 总体方案与架构 |
| `docs/识别打靶操作手册.md` | 视觉检测调试 |
| `docs/建图操作手册.md` | Cartographer 建图与定位 |
| `docs/外设使用注意事项.md` | 硬件接线与 udev |
| `docs/相机调用注意事项.md` | 相机 topic 与驱动 |
