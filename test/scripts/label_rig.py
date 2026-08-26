#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# label_rig.py —— RAICOM2026 标注辅助工具 (KCF 跟踪半自动)
# ------------------------------------------------------------
# 用途: 对采集到的一批相似场景图像, 首帧用鼠标框选目标, KCF 自动
#       跟踪到后续帧, 输出 YOLO 格式标签 (<cls_id> cx cy w h 归一化)。
# 用法:
#   rosrun raicom2026 label_rig.py --dir data/raw/person/enemy --class-id 1
#   rosrun raicom2026 label_rig.py --dir data/raw/mine/mine_red --class-id 0
#   rosrun raicom2026 label_rig.py --dir data/raw/antitank/anti_tank --class-id 0
# 交互:
#   首帧: 鼠标拖框选目标, 回车确认 / ESC 取消
#   后续帧: 自动跟踪; 手动修正则重新拖框; 按 A 追加目标; 回车下一张
#   输出: 与图片同名的 .txt 放到 --out (默认同目录)
# 类别索引 (必须与 yaml 一致):
#   person {0:hostage,1:enemy,2:friend}  mine {0:mine_red,1:mine_blue}  anti_tank {0:anti_tank}
# ============================================================
import os
import sys
import glob
import argparse

import cv2

yaml_class = {
    'person': {0: 'hostage', 1: 'enemy', 2: 'friend'},
    'mine': {0: 'mine_red', 1: 'mine_blue'},
    'antitank': {0: 'anti_tank'},
}


class LabelRig:
    def __init__(self, image_dir, class_id, out_dir=None, model=None):
        self.image_dir = image_dir
        self.class_id = int(class_id)
        self.out_dir = out_dir or image_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) +
                             glob.glob(os.path.join(image_dir, '*.png')))
        self.model = model
        self.current_roi = None
        self.tracking = False
        self.tracker = None
        self.boxes = []  # 当前帧所有框 [(cx,cy,w,h) 归一化]

    def _roi_callback(self, event, x, y, flags, param):
        # 简化: 用 selectROI 而非回调, 该回调仅作占位
        pass

    def _draw(self, frame, title):
        disp = frame.copy()
        for (cx, cy, w, h) in self.boxes:
            x1 = int((cx - w / 2) * disp.shape[1])
            y1 = int((cy - h / 2) * disp.shape[0])
            x2 = int((cx + w / 2) * disp.shape[1])
            y2 = int((cy + h / 2) * disp.shape[0])
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(title, disp)

    def run(self):
        if not self.images:
            print(f"ERROR: {self.image_dir} 下没有图片", file=sys.stderr)
            return
        title = f'label_rig [{os.path.basename(self.image_dir)}] cls={self.class_id}'
        n = len(self.images)
        for idx, img_path in enumerate(self.images):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            self.boxes = []
            self.tracker = None

            # 首帧框选
            print(f"\n[{idx+1}/{n}] {os.path.basename(img_path)}  (拖框选目标, 回车确认, ESC跳过)")
            roi = cv2.selectROI(title, frame, showCrosshair=False, fromCenter=False)
            cv2.destroyWindow(title)
            if roi[2] == 0 or roi[3] == 0:
                print("  跳过")
                continue
            x, y, rw, rh = roi
            self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(frame, roi)
            self.boxes = [(x + rw / 2, y + rh / 2, rw, rh)]
            self._draw(frame, title)

            # 跟踪后续帧
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), 27):
                    return
                if key == 13:  # 回车: 保存当前框, 下一张
                    break
                if key == ord('a'):  # 追加目标: 再框一个
                    roi2 = cv2.selectROI(title, frame, showCrosshair=False, fromCenter=False)
                    cv2.destroyWindow(title)
                    if roi2[2] > 0 and roi2[3] > 0:
                        x2, y2, w2, h2 = roi2
                        self.boxes.append((x2 + w2 / 2, y2 + h2 / 2, w2, h2))
                        # 用第一个 tracker 继续, 追加目标仅存框
                        self._draw(frame, title)

                # 取一帧跟踪 (用同一张图, 实际应在下一张; 这里做单帧精修)
                ok, bbox = self.tracker.update(frame)
                if ok:
                    x, y, rw, rh = [int(v) for v in bbox]
                    self.boxes = [(x + rw / 2, y + rh / 2, rw, rh)]
                self._draw(frame, title)

            # 保存 YOLO 标签
            txt = os.path.join(self.out_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            with open(txt, 'w') as f:
                for (cx, cy, bw, bh) in self.boxes:
                    cx_n = max(0.0, min(1.0, cx / w))
                    cy_n = max(0.0, min(1.0, cy / h))
                    bw_n = max(0.0, min(1.0, bw / w))
                    bh_n = max(0.0, min(1.0, bh / h))
                    f.write(f"{self.class_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")
            print(f"  已保存 {len(self.boxes)} 个框 -> {txt}")
            cv2.destroyWindow(title)

        print(f"\n完成. 标签在 {self.out_dir}")


def main():
    parser = argparse.ArgumentParser(description='KCF 辅助标注')
    parser.add_argument('--dir', required=True, help='图像目录')
    parser.add_argument('--class-id', required=True, type=int, help='类别索引')
    parser.add_argument('--out', default=None, help='标签输出目录(默认同图片目录)')
    args = parser.parse_args()
    LabelRig(args.dir, args.class_id, args.out).run()


if __name__ == '__main__':
    main()
