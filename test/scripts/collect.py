#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# collect.py —— RAICOM2026 数据采集工具 (三模型双相机, 自动打开)
# ------------------------------------------------------------
# 三个模型写在一起, 一台命令全部采集:
#   C    切换模型: person → mine → antitank → person
#        (person 用 webcam 848x480 MJPG; mine/antitank 用 orbbec)
#   Tab  切换当前模型内的类别
#   SPACE 连拍 burst 张 (默认 5) | S 单张 | Q/ESC 退出
#
# orbbec 自动打开:
#   1) 若 ROS master 未运行 → 自动拉起 roscore
#   2) 若 /camera/color/image_raw 无帧 → 自动 roslaunch
#      orbbec_camera dabai_dcw2.launch (DCW2 专用, enable_color 默认 true;
#      只开彩色, 关 depth/ir/点云, 省 USB 带宽)
#   3) 退出时自动关掉本脚本拉起的 roslaunch / roscore (不影响已在跑的节点)
#
# 用法:
#   rosrun raicom2026 collect.py                    # 默认 person 起步, 双相机都开
#   rosrun raicom2026 collect.py --mode mine        # 起步在 mine
#   rosrun raicom2026 collect.py --no-orbbec        # 不开 orbbec, 只 webcam
# 输出: data/raw/<模型>/<类别>/<timestamp>_<tag>.jpg + .json
# 注意: 只动 raicom2026/data/, 不碰其他包。
# ============================================================
import os
import sys
import time
import json
import socket
import signal
import datetime
import argparse
import subprocess

# 重定向到文件时也实时刷新输出, 方便看日志
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def _sigterm_to_interrupt(signum, frame):
    """把 SIGTERM 转成 KeyboardInterrupt, 走 run() 的 finally 统一清理
    (避免 timeout/kill 直接杀掉进程时泄漏拉起的 roscore / orbbec roslaunch)。"""
    raise KeyboardInterrupt

# ---- 环境兼容补丁 ----
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.insert(0, ros_path)

import cv2

PACKAGE_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
RAW_ROOT = os.path.join(PACKAGE_DATA, 'raw')

# 每个模型的 (相机, 类别列表); ORDER 决定 C 键循环顺序
MODELS = {
    'person':   {'camera': 'webcam', 'classes': ['enemy', 'hostage', 'friend', 'empty']},
    'mine':     {'camera': 'orbbec', 'classes': ['mine_red', 'mine_blue']},
    'antitank': {'camera': 'orbbec', 'classes': ['anti_tank']},
}
ORDER = ['person', 'mine', 'antitank']

# orbbec 自动打开的 launch 命令。
# 注意: 必须用 dabai_dcw2.launch (DCW2 专用, enable_color 默认 true)。
#      dabai_dw.launch 默认关彩色流, 节点会 "Stream color is disabled" 不出帧。
# 只开彩色 (640x480 MJPG), 关 depth/ir/点云, 省 USB2 带宽。
ORBBEC_LAUNCH = ('roslaunch orbbec_camera dabai_dcw2.launch '
                 'enable_depth:=false enable_ir:=false enable_point_cloud:=false')
ORBBEC_TOPIC = '/camera/color/image_raw'
ORBBEC_WAIT = 20          # 等第一帧的最大秒数
ORBBEC_LOG = '/tmp/collect_orbbec_roslaunch.log'


def make_dir_for(model, cls):
    d = os.path.join(RAW_ROOT, model, cls)
    os.makedirs(d, exist_ok=True)
    return d


class Collector:
    def __init__(self, start_mode='person', burst=5, auto_orbbec=True):
        self.burst = burst
        self.start_mode = start_mode
        self.mode = start_mode
        self.current_class = 0
        self.saved = 0
        self.save_root = RAW_ROOT

        # 两个相机状态
        self.webcap = None
        self.webcam_ok = False
        self.webcam_label = None

        self._rospy = None
        self._bridge = None
        self._orbbec_frame = None
        self.orbbec_ok = False
        self.orbbec_msg = 'orbbec 未启动'

        self._roscore_proc = None
        self._roslaunch_proc = None

        # 信号处理移到 run() 里注册: rospy.init_node() 会覆盖 SIGTERM/SIGINT,
        # 必须在 rospy 初始化完成后重新注册才会生效。

        self._open_webcam()
        if auto_orbbec:
            self._ensure_orbbec()

        if not self.webcam_ok and not self.orbbec_ok:
            print("ERROR: webcam 和 orbbec 都打不开", file=sys.stderr)
            sys.exit(1)

        if self.mode not in MODELS:
            self.mode = ORDER[0]
        if self.mode not in MODELS:
            print("ERROR: 无可用相机且无有效模式", file=sys.stderr)
            sys.exit(1)

    # ---------- webcam ----------
    def _open_webcam(self, retries=3):
        dev = '/dev/video0'
        for attempt in range(retries):
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.2)   # 等驱动起流, 避免首次 read 超时
            ok, frame = cap.read()
            if ok and frame is not None:
                self.webcap = cap
                self.webcam_ok = True
                h, w = frame.shape[:2]
                self.webcam_label = f'webcam_{w}x{h}'
                print(f"[collect] webcam 打开成功 {w}x{h} MJPG")
                return
            # 失败: 释放后短暂等待再重试 (设备偶尔瞬时失败)
            print(f"WARN: webcam 第{attempt+1}次取帧失败, 重试...", file=sys.stderr)
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(0.5)
        print("WARN: webcam 多次重试仍取不到帧, 仅 orbbec 模式可用", file=sys.stderr)
        if self.mode == 'person':
            self.mode = 'mine'

    # ---------- orbbec 自动打开 ----------
    def _master_alive(self):
        """探测 ROS master 是否在线。"""
        try:
            uri = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
            host = uri.replace('http://', '').replace('https://', '')
            if ':' in host:
                host, _, port = host.rpartition(':')
            else:
                port = '11311'
            s = socket.create_connection((host, int(port)), timeout=1.0)
            s.close()
            return True
        except Exception:
            return False

    def _start_roscore(self):
        print("[collect] ROS master 未运行, 自动拉起 roscore ...")
        try:
            self._roscore_proc = subprocess.Popen(
                ['roscore'], start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t0 = time.time()
            while time.time() - t0 < 10 and not self._master_alive():
                time.sleep(0.2)
            if self._master_alive():
                print("[collect] roscore 已就绪")
                return True
            print("WARN: roscore 启动超时", file=sys.stderr)
        except Exception as e:
            print(f"WARN: 自动拉起 roscore 失败: {e}", file=sys.stderr)
        return False

    def _ensure_orbbec(self):
        """保证 /camera/color/image_raw 有帧: 拉 master → 起 launch → 订阅。"""
        # 1) master
        if not self._master_alive():
            if not self._start_roscore():
                self.orbbec_msg = 'roscore 启动失败'
                return
        # 2) 订阅
        try:
            import rospy
            from sensor_msgs.msg import Image
            # 关键: cv_bridge 必须用 python3 构建版, 且要提到路径最前,
            # 否则会解析到 melodic 的 python2 cv_bridge (boost 模块 ABI 不兼容, 转换必失败)
            sys.path.insert(0, '/home/mowen/cv_bridge_ws/devel/lib/python3/dist-packages')
            from cv_bridge import CvBridge
            rospy.init_node('collect_dual', anonymous=True)
            self._rospy = rospy
            self._bridge = CvBridge()
            rospy.Subscriber(ORBBEC_TOPIC, Image, self._orbbec_cb, queue_size=1)
        except ImportError as e:
            self.orbbec_msg = f'ros 组件导入失败: {e}'
            return

        # 3) 等已有节点发帧
        t0 = time.time()
        while self._orbbec_frame is None and time.time() - t0 < 2 and not self._rospy.is_shutdown():
            time.sleep(0.05)
        if self._orbbec_frame is not None:
            self._finish_orbbec('已有 orbbec 节点')
            return

        # 4) 无节点 → 自动 roslaunch
        if not self._roslaunch_proc or self._roslaunch_proc.poll() is not None:
            print(f"[collect] 自动启动: {ORBBEC_LAUNCH}")
            env = os.environ.copy()
            # 请先 source ~/1raicom_ws/scripts/setup_env.sh
            try:
                logf = open(ORBBEC_LOG, 'w')
                self._roslaunch_proc = subprocess.Popen(
                    ORBBEC_LAUNCH.split(), start_new_session=True,
                    stdout=logf, stderr=subprocess.STDOUT, env=env)
            except Exception as e:
                self.orbbec_msg = f'orbbec 自动启动失败: {e}'
                return
            print(f"[collect] orbbec 启动中, 日志: {ORBBEC_LOG}")

        # 5) 等第一帧
        t0 = time.time()
        while (self._orbbec_frame is None and time.time() - t0 < ORBBEC_WAIT
               and not self._rospy.is_shutdown()):
            time.sleep(0.1)
            if self._roslaunch_proc and self._roslaunch_proc.poll() is not None:
                break
        if self._orbbec_frame is not None:
            self._finish_orbbec('orbbec 自动打开成功')
        else:
            self.orbbec_msg = 'orbbec 无帧(设备未插/占用/供电不足?)'
            print(f"WARN: {self.orbbec_msg} (日志见 {ORBBEC_LOG})", file=sys.stderr)

    def _finish_orbbec(self, msg):
        h, w = self._orbbec_frame.shape[:2]
        self.orbbec_ok = True
        self.orbbec_msg = f'{msg} {w}x{h}'
        print(f"[collect] {self.orbbec_msg}")

    def _orbbec_cb(self, msg):
        try:
            self._orbbec_frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    # ---------- 取帧/保存 ----------
    def active_camera(self):
        return MODELS[self.mode]['camera']

    def read_active_frame(self):
        """取当前模式对应相机的帧; 不可用返回 None。"""
        cam = self.active_camera()
        if cam == 'webcam':
            if not self.webcam_ok or self.webcap is None:
                return None
            ok, frame = self.webcap.read()
            return frame if ok else None
        else:
            if not self.orbbec_ok:
                return None
            return self._orbbec_frame

    def save(self, frame, tag):
        cls = MODELS[self.mode]['classes'][self.current_class]
        d = make_dir_for(self.mode, cls)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base = os.path.join(d, f'{ts}_{tag}')
        img_path = base + '.jpg'
        json_path = base + '.json'
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta = {
            'timestamp': ts,
            'camera': self.active_camera(),
            'model': self.mode,
            'class': cls,
            'width': frame.shape[1],
            'height': frame.shape[0],
            'tag': tag,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return img_path

    # ---------- 主循环 ----------
    def run(self):
        # rospy.init_node() 在 __init__ 里覆盖了信号处理, 这里重新注册
        # 让 SIGINT/SIGTERM 都能走 finally -> shutdown() 清理子进程
        signal.signal(signal.SIGTERM, _sigterm_to_interrupt)
        signal.signal(signal.SIGINT, _sigterm_to_interrupt)

        cls = MODELS[self.mode]['classes'][self.current_class]
        cam = self.active_camera()
        print(f"[collect] 起步 model={self.mode} camera={cam}")
        print(f"  C=切换模型 Tab=切换类别 SPACE=连拍{self.burst}张 S=单张 Q=退出")
        warn_since = 0.0

        try:
            while True:
                frame = self.read_active_frame()

                if frame is None:
                    # 当前相机不可用: 显示状态提示, 不退出
                    now = time.time()
                    if now - warn_since > 2.0:
                        print(f"WARN: {self.mode} 相机无帧 ({self.orbbec_msg})", file=sys.stderr)
                        warn_since = now
                    cv2.waitKey(50)
                    continue

                overlay = frame.copy()
                cv2.putText(overlay, f"[{self.mode}/{cls}] cam={cam}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(overlay,
                            f"saved={self.saved} C=model Tab=class SPACE={self.burst} S=1 Q=quit",
                            (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow('collect', overlay)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord(' '):
                    for i in range(self.burst):
                        f = self.read_active_frame()
                        if f is not None:
                            self.save(f, f'burst{i}')
                            self.saved += 1
                    print(f"  连拍 {self.burst} 张 -> {self.mode}/{cls} (累计 {self.saved})")
                elif key in (ord('s'), ord('S')):
                    p = self.save(frame, 'single')
                    self.saved += 1
                    print(f"  单张 {p}")
                elif key == 9:  # Tab: 模型内切类别
                    classes = MODELS[self.mode]['classes']
                    self.current_class = (self.current_class + 1) % len(classes)
                    cls = classes[self.current_class]
                    print(f"  切类别 -> {self.mode}/{cls}")
                elif key in (ord('c'), ord('C')):  # C: 切换模型
                    self.mode = ORDER[(ORDER.index(self.mode) + 1) % len(ORDER)]
                    self.current_class = 0
                    cls = MODELS[self.mode]['classes'][self.current_class]
                    cam = self.active_camera()
                    print(f"  切模型 -> {self.mode}/{cls} (cam={cam}, orbbec={'OK' if self.orbbec_ok else 'NO'})")
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            self.shutdown()

    def shutdown(self):
        """关闭本脚本拉起的 roslaunch/roscore (不影响已存在的节点)。"""
        for name, proc in (('orbbec roslaunch', self._roslaunch_proc),
                           ('roscore', self._roscore_proc)):
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    proc.wait(timeout=5)
                    print(f"[collect] 已停止本脚本拉起的 {name}")
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        if self.webcap is not None:
            self.webcap.release()
        print(f"[collect] 完成, 共保存 {self.saved} 张到 {RAW_ROOT}")


def main():
    parser = argparse.ArgumentParser(description='RAICOM2026 数据采集 (三模型双相机)')
    parser.add_argument('--mode', default='person', choices=ORDER,
                        help='起步模型, 之后按 C 循环切换')
    parser.add_argument('--burst', type=int, default=5, help='连拍张数')
    parser.add_argument('--no-orbbec', action='store_true', help='不自动打开 orbbec')
    args = parser.parse_args()
    Collector(args.mode, args.burst, auto_orbbec=not args.no_orbbec).run()


if __name__ == '__main__':
    main()
