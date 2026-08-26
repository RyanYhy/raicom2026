#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# eval_count.py —— RAICOM2026 计数评估 + 标注图保存工具
# ------------------------------------------------------------
# 功能1: 对一组"场景图片"(同一场景多帧) 做计数召回/精度评估。
#        这是比赛人物/地雷计数的离线验证, 目标: 运行时 conf 下
#        计数召回 >= 0.95。
# 功能2: 生成带标注框+类别+置信度的保存图 —— 即比赛要求的
#        "识别出的图片保存至本地" 的离线验证版。
# 用法:
#   rosrun raicom2026 eval_count.py --model models/person.onnx \
#       --names "{0:hostage,1:enemy,2:friend}" --dir data/raw/person/enemy \
#       --truth 1   (可选: 每张真值计数, 无则仅出图)
#   rosrun raicom2026 eval_count.py --model models/mine.onnx \
#       --names "{0:mine_red,1:mine_blue}" --dir data/raw/mine/mine_red \
#       --truth 1 --out results/
# 输出: 每张图 annotated_<原文件名>.jpg 到 --out; 统计到 stdout
# ============================================================
import os
import sys
import glob
import json
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from raicom2026.yolo_onnx import YOLOv8ONNX, draw_detections


def parse_names(s):
    """解析 '0:hostage,1:enemy,2:friend' -> {0:'hostage',...}"""
    names = {}
    for part in s.split(','):
        part = part.strip()
        if ':' in part:
            k, v = part.split(':', 1)
            names[int(k.strip())] = v.strip()
    return names


def main():
    parser = argparse.ArgumentParser(description='计数评估 + 标注图保存')
    parser.add_argument('--model', required=True, help='onnx 模型路径')
    parser.add_argument('--names', default='0:a', help='类别字典 0:name,1:name')
    parser.add_argument('--dir', required=True, help='图片目录')
    parser.add_argument('--truth', type=int, default=None,
                        help='每张图真值计数(可选); 提供则输出召回/精度')
    parser.add_argument('--conf', type=float, default=0.30, help='运行时置信度阈值')
    parser.add_argument('--out', default='results', help='标注图输出目录')
    parser.add_argument('--font', default=None, help='中文字体路径(可选)')
    args = parser.parse_args()

    names = parse_names(args.names)
    # 反向映射 name->cls_id (标注颜色用)
    name_to_id = {v: k for k, v in names.items()}

    model = YOLOv8ONNX(args.model, conf_thres=args.conf, iou_thres=0.45)
    os.makedirs(args.out, exist_ok=True)

    images = sorted(glob.glob(os.path.join(args.dir, '*.jpg')) +
                    glob.glob(os.path.join(args.dir, '*.png')))
    if not images:
        print(f"ERROR: {args.dir} 无图片", file=sys.stderr)
        sys.exit(1)

    total_tp = total_fn = total_fp = 0
    total_truth = 0
    print(f"模型: {args.model} conf={args.conf}")
    print(f"评估 {len(images)} 张 -> {args.out}")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        boxes, scores, class_ids = model.detect(img)
        # 标注图
        annotated = draw_detections(img, boxes, scores, class_ids, names)
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.out, f'annotated_{base}.jpg')
        cv2.imwrite(out_path, annotated)

        if args.truth is not None:
            # 计数评估: 用检测框数量 vs 真值
            det_count = len(class_ids)
            tp = min(det_count, args.truth)
            fn = args.truth - tp
            fp = det_count - tp
            total_tp += tp
            total_fn += fn
            total_fp += fp
            total_truth += args.truth

    if args.truth is not None:
        recall = total_tp / max(1, total_truth)
        precision = total_tp / max(1, total_tp + total_fp)
        print(f"\n=== 计数评估 (真值每张 {args.truth}) ===")
        print(f"TP={total_tp} FN={total_fn} FP={total_fp} 真值总数={total_truth}")
        print(f"召回率 Recall = {recall:.3f} (目标 >= 0.95)")
        print(f"精度 Precision = {precision:.3f}")
    print(f"标注图已保存到 {args.out}/annotated_*.jpg")


if __name__ == '__main__':
    main()
