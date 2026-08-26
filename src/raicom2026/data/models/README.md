# 已训练模型（2026-08-09 归位并验证）

> 来源：`data/tools/`（训练输出，2026-08-09 复制归位）
> 验证：onnxruntime 读入，输入 640×640，输出 `[1, 4+nc, 8400]`，与 `src/raicom2026/yolo_onnx.py` 的 NMS 匹配 ✅

## 模型与类别映射（⚠️ 索引顺序必须与后处理字典一致）

| 文件 | 输入 | nc | 类别索引 |
|---|---|---|---|
| `person.onnx` | 640×640 | 3 | `{0: hostage, 1: enemy, 2: friend}` |
| `mine.onnx` | 640×640 | 2 | `{0: mine_red, 1: mine_blue}`（红=未排雷，蓝=已排雷） |
| `antitank.onnx` | 640×640 | 1 | `{0: anti_tank}` |

## 调用约定

- 运行时节点只依赖 onnxruntime（yolov8 conda 环境），**不 import torch/ultralytics**
- 共享推理：`from raicom2026.yolo_onnx import YOLOv8ONNX`（letterbox 预处理 + NMS + 标注图）
- `fire.onnx` 已废弃（target 类不存在），打靶目标用 person.onnx 的 **enemy(1)** 类
- 旧 `cv_bridge_ws/person.onnx` 是 2 类旧版，勿用

## 验证建议（下一步）

```bash
# 每个模型跑一遍计数召回评估, 目标 conf 下召回 >= 0.95
rosrun raicom2026 eval_count.py --model data/models/person.onnx \
    --names "0:hostage,1:enemy,2:friend" --dir data/eval/person_scene --truth 3 --out results/
```
