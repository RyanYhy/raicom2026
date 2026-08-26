# -*- coding: utf-8 -*-

# ==========================================
# ！！！ 环境兼容性补丁 (必须放在最前面) ！！！
# ==========================================
import sys
import os
import ctypes

ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.insert(0, ros_path)

try:
    ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libffi.so.7")
    #ROS环境特殊性：ROS的Python环境可能带有自定义编译的扩展模块
    #解决Python扩展模块的依赖问题
except OSError:
    pass 

try:
    from cv_bridge import CvBridge, CvBridgeError
    import cv_bridge.boost.cv_bridge_boost
except ImportError:
    pass
# ==========================================

import rospy
import cv2
import math
import numpy as np
import onnxruntime as ort
import time
import atexit
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerRequest

# 新增：引入 Ultralytics 用于车辆检测
try:
    from ultralytics import YOLO
except ImportError:
    rospy.logerr("缺少 ultralytics 库，请先安装: pip install ultralytics")

# --- NMS辅助函数 ---
def calculate_iou(box, other_boxes):
    x1 = np.maximum(box[0], np.array(other_boxes)[:, 0])
    y1 = np.maximum(box[1], np.array(other_boxes)[:, 1])
    x2 = np.minimum(box[0] + box[2], np.array(other_boxes)[:, 0] + np.array(other_boxes)[:, 2])
    y2 = np.minimum(box[1] + box[3], np.array(other_boxes)[:, 1] + np.array(other_boxes)[:, 3])
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = box[2] * box[3]
    other_boxes_area = np.array(other_boxes)[:, 2] * np.array(other_boxes)[:, 3]
    iou = intersection_area / (box_area + other_boxes_area - intersection_area)
    return iou

def custom_NMSBoxes(boxes, scores, confidence_threshold, iou_threshold):
    if len(boxes) == 0: return []
    scores = np.array(scores)
    boxes = np.array(boxes)
    mask = scores > confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    original_indices = np.arange(len(boxes))[mask]

    if len(filtered_boxes) == 0: return []
    
    sorted_indices = np.argsort(filtered_scores)[::-1]
    keep_indices = []
    
    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        keep_indices.append(current_index)
        if len(sorted_indices) == 1: break
        current_box = filtered_boxes[current_index]
        other_boxes = filtered_boxes[sorted_indices[1:]]
        iou = calculate_iou(current_box, other_boxes)
        non_overlapping_indices = np.where(iou <= iou_threshold)[0]
        sorted_indices = sorted_indices[non_overlapping_indices + 1]
    
    return original_indices[keep_indices].tolist()

# --- YOLOv8 ONNX 推理封装类 (用于其他轻量级任务) ---
class YOLOv8ONNX:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model_path = model_path

        if not os.path.exists(model_path):
            rospy.logerr(f"Error: Model file '{model_path}' not found.")
            self.session = None
            return

        providers = ['CPUExecutionProvider']
        #ONNX Runtime 的执行提供者配置，用于指定模型在哪个硬件上运行
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            rospy.loginfo(f"ONNX 模型加载成功: {model_path}")
        except Exception as e:
            rospy.logerr(f"ONNX 模型加载失败: {e}")
            self.session = None
            return

        self.model_inputs = self.session.get_inputs()   # 获取模型输入信息
        self.input_shape = self.model_inputs[0].shape   # 获取第一个输入的形状
        self.input_width = self.input_shape[2]  # 输入图片的宽度
        self.input_height = self.input_shape[3] # 输入图片的高度

    def preprocess(self, img):
        self.img_height, self.img_width = img.shape[:2]     # 保存原始图片尺寸
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height)) # resize=img /原始1 * 规定
        image_data = np.array(img) / 255.0  # 像素值归一化
        # 将像素值从[0,255]范围缩放到[0,1]范围，有助于模型训练和推理的稳定性
        image_data = np.transpose(image_data, (2, 0, 1))    # 调整维度顺序
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)  # 添加batch维度
        # 批量图片 (有batch维度)
        return image_data

    def postprocess(self, output):
        outputs = np.transpose(np.squeeze(output[0]))   # 去掉第一个（batch）维度
        # 作用：将模型输出的3D张量转换为2D矩阵，方便按预测框处理
        # 原始格式: (1, features, 8400) → 转置后: (8400, features)
        rows = outputs.shape[0]
        # 获取预测框总数
        boxes, scores, class_ids = [], [], []
        # 用于存储筛选后的结果

        # 计算坐标缩放因子，乘完后复原成原坐标
        x_factor = self.img_width / self.input_width    #原始/规定
        y_factor = self.img_height / self.input_height


        for i in range(rows):
            # 提取当前预测框的类别分数部分，前4个是中心点x、y坐标，宽度，高度
            classes_scores = outputs[i][4:]
            # 找到最高的类别分数
            max_score = np.amax(classes_scores)
            if max_score >= self.conf_thres:
                # np.argmax: 返回最大值所在的索引
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # 转换1: 中心点格式 → 左上角格式
                # 转换2: 模型输入尺寸 → 原始图片尺寸
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 应用非极大值抑制(NMS)，作用：去除重叠的检测框，保留最佳的一个
        indices = custom_NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        #conf_thres: 置信度阈值  iou_thres: IOU阈值，用于判断两个框是否重叠
        final_boxes, final_scores, final_class_ids = [], [], []
        for i in indices:
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])
        return final_boxes, final_scores, final_class_ids

    def detect(self, image):
        if self.session is None: return [], [], []
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.model_inputs[0].name: input_tensor})     # 执行模型推理
        # 语法：session.run(output_names, input_feed)
        # output_names=None：表示获取所有输出
        # input_feed={input_name: input_tensor}:
        # 这是一个字典，将输入节点的名称映射到输入数据

        return self.postprocess(outputs)


# --- 核心多任务检测类 ---
class MultiTaskDetector:
    def __init__(self):
        rospy.init_node('multi_task_detector', anonymous=True)  #anonymous=True会自动在名称后添加随机数，确保唯一性
        self.bridge = CvBridge()
        self.current_task = None 
        self.last_process_time = 0
        
        # 全局文件对象（追加模式）用于记录日志
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        log_file_path = os.path.join(log_dir, "gunclient_output.txt")
        self.global_file = open(log_file_path, "a", buffering=1, encoding="utf-8")
        atexit.register(self.global_file.close)
        
        # 记录程序启动
        self.write_to_file("****************************New run****************************** ")
        
        #超时处理
        #！！！！！需注意设置的超时时间要比nav_competition中的短
        self.task_start_time = {}
        self.task_timeouts = {  # 各任务超时时间
            'fire': 25.0,    # 对应people_timeout
            'person': 8.0,  # 对应fire_timeout
        }

        # --- 配置: 图片保存 ---
        current_dir = os.path.dirname(os.path.abspath(__file__))    #__file__是Python内置变量 #os.path.abspath是去掉文件名的路径
        self.save_dir = os.path.join(current_dir, "save_img")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            rospy.loginfo(f"创建保存目录: {self.save_dir}")
        
        # --- 字体加载 ---
        self.font_path = self._find_chinese_font()
        rospy.loginfo(f"使用字体: {self.font_path}")

        # ================== 模型加载 ==================
        rospy.loginfo(">>> [Init] 加载模型中...")

        #conf_thres：控制"检测灵敏度"，平衡漏检和误检
        #iou_thres：控制"去重强度"，平衡检测精度和重复框

        # 1. 打靶
        self.fire_model = YOLOv8ONNX("fire.onnx", conf_thres=0.75, iou_thres=0.5)
        self.fire_classes = {0: "target"}
        
        # 2. 人偶
        self.person_model = YOLOv8ONNX("person.onnx", conf_thres=0.50, iou_thres=0.45)
        self.person_classes = {0: "hostage", 1: "enemy", 2: "friend"}

        # /gun 服务相关标志
        self.gun_service_available = False
        self.gun_service_called = False  # 标记服务是否已调用过
        self.gun_completed = False

        # ================== 通信接口 ==================
        # 其他callback里改变 current_task!
        self.image_subscriber = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)  # buff_size 是 ROS Subscriber 的缓冲区大小
        
        self.fire_cmd_sub = rospy.Subscriber('/fire', String, self.fire_cmd_callback)
        self.fire_res_pub = rospy.Publisher('/fire_shoot_status', String, queue_size=10)
        
        self.person_cmd_sub = rospy.Subscriber('/person_detect', String, self.person_cmd_callback)
        self.person_res_pub = rospy.Publisher('/person_detect_status', String, queue_size=10)

        rospy.loginfo("=== 多任务检测节点已就绪 (包含追踪与快照功能) ===")

    def write_to_file(self, data):
        """向全局文件写入数据"""
        self.global_file.write(f"{data}\n")
        self.global_file.flush()  # 立即刷新缓冲区
        rospy.loginfo(f"日志记录: {data}")

    def call_gun_service(self):
        """调用 /gun 服务"""
        try:
            # 如果服务已调用过，直接返回成功
            if self.gun_service_called:
                self.write_to_file("/gun 服务已调用过，跳过重复调用")
                return True
                
            # 等待服务可用
            self.write_to_file("等待 /gun 服务...")
            rospy.wait_for_service('/gun')
            
            try:
                # 创建服务代理
                gun_service = rospy.ServiceProxy('/gun', Trigger)
                
                # 调用服务
                self.write_to_file("正在调用 /gun 服务...")
                response = gun_service()
                
                # 处理响应
                if response.success:
                    self.write_to_file(f"服务调用成功: {response.success}")
                    self.write_to_file(f"返回消息: {response.message}")

                    
                    # 标记服务已调用
                    self.gun_service_called = True
                    self.gun_service_available = True
                    if response.message.strip() == "done":
                        self.write_to_file("收到'done'消息")
                        self.gun_completed = True
                    return True
                else:
                    self.write_to_file(f"服务调用失败: {response.message}")
                    return False
                    
            except rospy.ServiceException as e:
                self.write_to_file(f"服务调用失败: {e}")
                return False
                
        except rospy.ROSException as e:
            self.write_to_file(f"等待服务时出错: {e}")
            return False

    # ================== 辅助函数 ==================
    def _find_chinese_font(self):
        # 预定义可能的中文字体路径列表
        possible_fonts = ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]
        for p in possible_fonts:
            if os.path.exists(p): return p
        return None

    def _save_image_with_timestamp(self, frame, prefix="result"):
        # 保存带时间戳的图片，防止覆盖
        import datetime

        # 确保目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        save_path = os.path.join(self.save_dir, filename)

        try:
            cv2.imwrite(save_path, frame)
            rospy.loginfo(f"图片已保存: {save_path}")
        except Exception as e:
            rospy.logerr(f"保存失败: {e}")

    def _check_task_timeout(self):
        # 检查当前任务是否超时
        if self.current_task is None:
            return False

        if self.current_task not in self.task_start_time:
            return False

        if self.current_task not in self.task_timeouts:
            rospy.logwarn(f"任务 {self.current_task} 没有设置超时时间")
            return False

        elapsed_time = time.time() - self.task_start_time[self.current_task]
        timeout = self.task_timeouts[self.current_task]

        if elapsed_time > timeout:
            rospy.logwarn(f"任务 {self.current_task} 已超时 ({elapsed_time:.1f}秒 > {timeout}秒)")
            return True

        return False

    def _force_end_current_task(self):
        """强制结束当前任务（超时）"""
        if self.current_task is None:
            return

        rospy.logwarn(f"强制结束任务: {self.current_task}")

        if self.current_task == 'fire':
            rospy.logwarn("打靶任务超时")
            # self.fire_res_pub.publish("fire_failed")
        elif self.current_task == 'person':
            rospy.logwarn("人偶识别超时")
            # msg = String()
            # msg.data = ""
            # self.person_res_pub.publish(msg)

        # 清理任务状态
        if self.current_task in self.task_start_time:
            del self.task_start_time[self.current_task]
        self.current_task = None

    # ================== 触发回调 ==================
    def fire_cmd_callback(self, msg):
        if msg.data: 
            self._switch_task('fire', "开始准备打靶...")
            # 重置服务调用标志，确保每次fire任务都可以调用服务
            self.gun_service_called = False
            self.gun_service_available = False
            self.gun_completed = False
    
    def person_cmd_callback(self, msg):
        if msg.data: 
            self._switch_task('person', "开始人偶识别...")

    def _switch_task(self, task_name, log_msg):
        if self.current_task != task_name:
            rospy.loginfo(f">>> [切换任务] {log_msg}")

            # 记录任务开始时间
            self.task_start_time[task_name] = time.time()

            self.current_task = task_name

            self.last_process_time = 0 
        else:
            rospy.loginfo(f">>> 正在进行 {task_name}，忽略指令")

    # ================== 图像处理主循环 ==================
    def image_callback(self, msg):
        if self.current_task is None: return

        if self._check_task_timeout():
            self._force_end_current_task()
            return

        # 速率控制：使用 1Hz 限流
        if time.time() - self.last_process_time < 1.0: return
        self.last_process_time = time.time()
            # 只有当间隔 ≥1秒时才继续处理

        try:
            # 将ROS的Image消息转换为OpenCV的numpy数组格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except (CvBridgeError, ImportError) as e:
            rospy.logerr(f"Image Error: {e}")
            return

        # 任务分发
        if self.current_task == 'fire': 
            self.process_fire_logic(cv_image)
        elif self.current_task == 'person': 
            self.process_person_logic(cv_image)
        

    # ================== 逻辑实现 ==================
    def process_fire_logic(self, frame):           
        # 调用 /gun 服务
        if self.call_gun_service():
            self.write_to_file("打靶服务调用成功，发布完成信号")
        else:
            self.write_to_file("打靶服务调用失败")
            # 即使服务调用失败，也发布完成信号，避免任务卡住
        
        if self.gun_completed == True:
            # 检测成功，结束任务
            if self.current_task in self.task_start_time:
                del self.task_start_time[self.current_task]
            self.current_task = None
    
        return


    def process_person_logic(self, frame):
        boxes, scores, class_ids = self.person_model.detect(frame)
        detected_targets = []
        if len(class_ids) > 0:
            for cls_id, score in zip(class_ids, scores):    # zip() 函数将两个列表组合成元组对
                cls_id = int(cls_id)
                if cls_id in self.person_classes:
                    target_name = self.person_classes[cls_id]
                    detected_targets.append(target_name)

        if detected_targets:
            # 将检测到的人偶名称连接成字符串
            final_result_str = ",".join(detected_targets)   # join() 方法用指定分隔符连接列表元素
            rospy.loginfo(f"SUCCESS: 识别到人偶: {final_result_str}")
            self._save_image_with_timestamp(frame, "result_person.jpg")
            #可以改文件名
            msg = String()
            msg.data = final_result_str
            self.person_res_pub.publish(msg)
            if self.current_task in self.task_start_time:
                del self.task_start_time[self.current_task]
            self.current_task = None
        else:
            rospy.loginfo("未检测到人偶，继续尝试...")
            return


if __name__ == '__main__':
    try:
        detector = MultiTaskDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass