#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import serial
import time
import struct
from threading import Thread
from geometry_msgs.msg import Twist

# 全局串口对象
ser = None

def cmd_vel_callback(msg):
    """
    接收 /cmd_vel 消息并将其转换为串口指令发送给底盘
    """
    global ser

    if ser is None or not ser.is_open:
        rospy.logwarn("Serial port not available, skipping cmd_vel message.")
        return

    # 将速度指令从 m/s 和 rad/s 转换为毫米/秒和毫弧度/秒
    vx = int(msg.linear.x * 1000)
    vy = 0
    vz = int(msg.angular.z * 1000)

    try:
        # 将速度值打包成2字节的小端有符号整数 (short)
        # '<' 表示小端, 'h' 表示有符号短整型 (2字节)
        vx_bytes = struct.pack('<h', vx)
        vy_bytes = struct.pack('<h', vy)
        vz_bytes = struct.pack('<h', vz)

        # 构建发送的数据帧
        # 协议格式: [帧头1, 帧头2, 功能码, 长度, 指令码, vx_low, vx_high, vy_low, vy_high, vz_low, vz_high, 校验和]
        frame = bytearray([
            0xAA, 0xBB, # 帧头
            0x0A,       # 功能码
            0x12,       # 长度
            0x02,       # 指令码
            vx_bytes[0], vx_bytes[1], # X速度 (低位, 高位)
            vy_bytes[0], vy_bytes[1], # Y速度 (低位, 高位)
            vz_bytes[0], vz_bytes[1], # Z角速度 (低位, 高位)
        ])

        # 计算校验和 (这里简单设置为0, 如果底层需要真实校验则需实现)
        checksum = 0x00
        frame.append(checksum)

        # 发送数据
        ser.write(frame)

    except Exception as e:
        rospy.logerr("Failed to send command to serial port: %s", e)


def read_serial_thread(vel_publisher):
    """
    在独立的线程中持续读取串口数据, 解析并发布为 vel_raw 话题
    """
    global ser

    while not rospy.is_shutdown() and ser.is_open:
        try:
            # 寻找帧头 0xAA 0xBB
            header = ser.read(2)
            if header != b'\xAA\xBB':
                continue

            # 读取剩余的数据 (10字节)
            data_bytes = ser.read(10)
            if len(data_bytes) < 10:
                rospy.logwarn("Incomplete serial frame received.")
                continue

            # ** 修正点 **
            # 直接从数据负载中解包速度值。
            # 原始数据中，速度信息从第3个字节开始(索引为2)。
            # 我们需要解包3个有符号短整型(hhh)，总共6个字节。
            vx, vy, vz = struct.unpack('<hhh', data_bytes[2:8])

            # 将读取到的整数值转换为 m/s 和 rad/s
            twist_msg = Twist()
            twist_msg.linear.x = vx / 1000.0
            twist_msg.linear.y = 0
            twist_msg.angular.z = (vz / 1000.0) *1.0 # 根据pubv.py中的系数

            vel_publisher.publish(twist_msg)

        except struct.error as e:
            rospy.logerr("Failed to unpack serial data: %s. Frame might be corrupted.", e)
        except Exception as e:
            rospy.logerr("Error while reading from serial: %s", e)
            time.sleep(1) # 发生错误时等待一下, 避免刷屏

def main():
    global ser
    rospy.init_node('base_controller')

    # 从参数服务器获取串口配置
    port = rospy.get_param('~port', '/dev/carserial')
    baud = rospy.get_param('~baud', 115200)

    try:
        ser = serial.Serial(port, baud, timeout=1)
        if ser.is_open:
            rospy.loginfo("Serial port %s opened successfully at %d baud.", port, baud)
    except serial.SerialException as e:
        rospy.logfatal("Failed to open serial port %s: %s", port, e)
        return

    # 创建话题发布者和订阅者 (使用相对名称, ROS会自动添加命名空间)
    vel_raw_pub = rospy.Publisher('vel_raw', Twist, queue_size=10)
    rospy.Subscriber('cmd_vel', Twist, cmd_vel_callback, queue_size=1)

    # 启动串口读取线程
    reader_thread = Thread(target=read_serial_thread, args=(vel_raw_pub,))
    reader_thread.daemon = True # 设置为守护线程, ROS关闭时自动退出
    reader_thread.start()
    
    rospy.loginfo("Base controller node started.")
    rospy.spin()

    # ROS关闭时, 清理资源
    if ser and ser.is_open:
        ser.close()
        rospy.loginfo("Serial port %s closed.", port)

if __name__ == '__main__':
    main()
