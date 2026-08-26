# raicom2026

RAICOM 2026「智能侦察」赛道专用包。**本包是唯一允许编写/修改代码的地方。**

## 🚫 行为准则（违反 = 不可逆后果）

1. **只动本包**：所有新代码、修改、配置、文档一律写在 `raicom2026/` 内。
2. **其他包只读**：`1205`、`robot_bringup` / `robot_navigation`、`my_local_planner` 等只能参考，**禁止修改**。
3. **禁止 rosbag**：比赛只允许 SLAM 激光建图导航 + 视觉导航，出现 `.bag` = 判 0 分。
4. **串口唯一**：`/dev/gun` 只由 `gimbal_server.py` 打开，其他进程走 `/gun/*` 服务。
5. **运行环境**：onnxruntime 节点跑 `yolov8` conda 环境（py3.8），运行时**不 import torch**。

## 📁 目录结构

```
raicom2026/
├── CMakeLists.txt      # catkin 构建配置（srv 生成 + 脚本安装）
├── package.xml         # 包元数据
├── setup.py            # python 包安装
├── README.md           # 本文件（准则）
├── docs/
│   └── 比赛方案与技术文档.md   # ★ 完整比赛方案 + 行为准则 + 开发计划
├── launch/             # 本包专属 launch（usb_cam 848×480、orbbec、比赛总装）
├── param/              # params.yaml、costmap、DWA、AMCL 参数
├── srv/                # GunAim.srv / GunFire.srv
├── scripts/            # 比赛节点（gimbal_server/webcam_person_gun/orbbec_mine_antitank/watchdog/nav）
├── src/raicom2026/  # python 模块
└── data/               # 训练数据、标注工具、模型
```

## 📄 方案

完整方案见 [docs/比赛方案与技术文档.md](docs/比赛方案与技术文档.md)
