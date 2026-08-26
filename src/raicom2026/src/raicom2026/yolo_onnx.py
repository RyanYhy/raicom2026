# -*- coding: utf-8 -*-
# ============================================================
# raicom2026/yolo_onnx.py —— YOLOv8 ONNX 推理共享模块
# ------------------------------------------------------------
# 所有比赛/工具节点共用的推理封装: letterbox 预处理 + NMS 后处理
# + 标注图绘制。只依赖 onnxruntime + numpy + opencv, 不 import torch。
# 类别字典必须与训练 yaml 索引一致。
# ============================================================
import os
import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img, new_shape=640, color=(114, 114, 114), auto=False, stride=32):
    """YOLOv8 标准 letterbox: 等比缩放 + 灰边填充, 不改纵横比。
    返回 (padded_img, ratio, (dw, dh))。"""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


class YOLOv8ONNX:
    def __init__(self, model_path, conf_thres=0.30, iou_thres=0.45,
                 intra_op_num_threads=2, letterbox_input=True):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.letterbox_input = letterbox_input
        self.model_path = model_path
        self.session = None
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_num_threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'], sess_options=so)
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]

    def preprocess(self, img):
        self.img_h, self.img_w = img.shape[:2]
        if self.letterbox_input:
            padded, self.ratio, (self.dw, self.dh) = letterbox(img, self.input_width)
        else:
            padded = cv2.resize(img, (self.input_width, self.input_height))
            self.ratio = (self.input_width / self.img_w, self.input_height / self.img_h)
            self.dw, self.dh = 0.0, 0.0
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img_data = np.ascontiguousarray(img_rgb, dtype=np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1))
        return np.expand_dims(img_data, axis=0)

    def postprocess(self, output):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        if self.letterbox_input:
            # letterbox 输入: 框在 640x640 填充坐标里, 必须先减去半填充(dw/dh)
            # 再除以缩放比, 才是原图坐标。只用缩放比/只乘因子都会把垂直坐标偏掉
            # (webcam 848x480 / orbbec 640x480 非正方形输入必须走这条路)。
            r = self.ratio
            dw, dh = self.dw, self.dh
        else:
            rw = self.img_w / self.input_width
            rh = self.img_h / self.input_height
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.conf_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                if self.letterbox_input:
                    left = int((x - w / 2 - dw) / r)
                    top = int((y - h / 2 - dh) / r)
                    width = int(w / r)
                    height = int(h / r)
                else:
                    left = int((x - w / 2) * rw)
                    top = int((y - h / 2) * rh)
                    width = int(w * rw)
                    height = int(h * rh)
                class_ids.append(int(class_id))
                scores.append(float(max_score))
                boxes.append([left, top, width, height])
        indices = self.nms(boxes, scores)
        final_boxes, final_scores, final_class_ids = [], [], []
        for i in indices:
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])
        return final_boxes, final_scores, final_class_ids

    def nms(self, boxes, scores):
        if not boxes:
            return []
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 0] + boxes[i, 2], boxes[order[1:], 0] + boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 1] + boxes[i, 3], boxes[order[1:], 1] + boxes[order[1:], 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = boxes[i, 2] * boxes[i, 3]
            area_o = boxes[order[1:], 2] * boxes[order[1:], 3]
            iou = inter / (area_i + area_o - inter + 1e-6)
            order = order[1:][iou <= self.iou_thres]
        return keep

    def detect(self, image):
        if self.session is None:
            return [], [], []
        tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
        return self.postprocess(outputs)


def draw_detections(img, boxes, scores, class_ids, class_names, color_map=None,
                    chinese_font=None):
    """在原图上画检测框+类别+置信度, 返回新图。"""
    out = img.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, w, h = box
        color = color_map.get(cls_id, (0, 255, 0)) if color_map else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x1 + w, y1 + h), color, 2)
        label = f"{class_names.get(cls_id, str(cls_id))} {score:.2f}"
        if chinese_font:
            font_path = chinese_font
            # opencv putText 不支持中文, 用 PIL 或跳过; 此处回退英文
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def find_chinese_font():
    for p in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
              '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf']:
        if os.path.exists(p):
            return p
    return None
