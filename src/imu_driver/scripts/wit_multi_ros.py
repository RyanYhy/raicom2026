#!/usr/bin/env python
# -*- coding:utf-8 -*-
import serial
import struct
import rospy
import time
import math
import sys
import platform
import threading
import serial.tools.list_ports
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from sensor_msgs.msg import NavSatFix
from tf.transformations import quaternion_from_euler


# 查找 ttyUSB* 设备        Look for ttyUSB* devices
def find_ttyUSB():
    print('The default serial port of the imu is /dev/ttyUSB0, if multiple serial port devices are identified, modify the serial port corresponding to the imu in the launch file')
    posts = [port.device for port in serial.tools.list_ports.comports() if 'USB' in port.device]
    print('There are {} {} serial port devices connected to the current PC: {}'.format(len(posts), 'USB', posts))


# 校验
def checkSum(list_data, check_data):
    return sum(list_data) & 0xff == check_data


# 16 进制转 ieee 浮点数
def hex_to_short(raw_data):
    return list(struct.unpack("hhhh", bytearray(raw_data)))

# GPS经纬度解析
def hex_to_data(raw_data):
    return list(struct.unpack("i", bytearray(raw_data)))
    
# GPS高度解析
def hex_to_altitude(raw_data):
    return list(struct.unpack("h", bytearray(raw_data)))
    
# 处理串口数据
def handleSerialData(raw_data):
    global buff, key, angle_degree, magnetometer, acceleration, angularVelocity, version, longitude_imu, latitude_imu, altitude_imu, frame_id
    angle_flag=False
    if python_version == '2':
        buff[key] = ord(raw_data)
    if python_version == '3':
        buff[key] = raw_data

    key += 1
    if buff[0] != 0x55:
        key = 0
        return
    if key < 11:  # 根据数据长度位的判断, 来获取对应长度数据
        return
    else:
        data_buff = list(buff.values())  # 获取字典所有 value
        if buff[1] == 0x51 :
            if checkSum(data_buff[0:10], data_buff[10]):
                acceleration = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 16 * 9.8 for i in range(0, 3)]
            else:
                print('0x51 Check failure')

        elif buff[1] == 0x52:
            if checkSum(data_buff[0:10], data_buff[10]):
                angularVelocity = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 2000 * math.pi / 180 for i in range(0, 3)]

            else:
                print('0x52 Check failure')

        elif buff[1] == 0x53:
            if checkSum(data_buff[0:10], data_buff[10]):
                temp = hex_to_short(data_buff[2:10])
                angle_degree = [temp[i] / 32768.0 * 180 for i in range(0, 3)]
                version = temp[3]
                angle_flag = True
            else:
                print('0x53 Check failure')
        elif buff[1] == 0x54:
            if checkSum(data_buff[0:10], data_buff[10]): 
                magnetometer = hex_to_short(data_buff[2:10])
            else:
                print('0x54 Check failure')

        elif buff[1] == 0x57:
            if checkSum(data_buff[0:10], data_buff[10]):
                longitude_imu = (hex_to_data(data_buff[2:6])[0]  // 10000000.0 * 100 ) +  ((hex_to_data(data_buff[2:6])[0]  % 10000000) / 10000000.0)
                latitude_imu = (hex_to_data(data_buff[6:10])[0]  // 10000000.0 * 100 ) +((hex_to_data(data_buff[6:10])[0] % 10000000) / 10000000.0)
            else:
                print('0x57 Check failure')

        elif buff[1] == 0x58:
            if checkSum(data_buff[0:10], data_buff[10]): 
                altitude_imu = hex_to_altitude(data_buff[2:4])[0]  / 10.0
            else:
                print('0x58 Check failure')

        else:
            buff = {}
            key = 0

        buff = {}
        key = 0
        if angle_flag:
            stamp = rospy.get_rostime()

            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = frame_id

            mag_msg.header.stamp = stamp
            mag_msg.header.frame_id = frame_id

            location_msg.header.stamp = stamp
            location_msg.header.frame_id = frame_id
            
            angle_radian = [angle_degree[i] * math.pi / 180 for i in range(3)]
            qua = quaternion_from_euler(angle_radian[0], angle_radian[1], angle_radian[2])

            imu_msg.orientation.x = qua[0]
            imu_msg.orientation.y = qua[1]
            imu_msg.orientation.z = qua[2]
            imu_msg.orientation.w = qua[3]
            imu_msg.orientation_covariance= [1e-3, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 1e-3]
            
            imu_msg.angular_velocity.x = angularVelocity[0]
            imu_msg.angular_velocity.y = angularVelocity[1]
            imu_msg.angular_velocity.z = angularVelocity[2]
            imu_msg.angular_velocity_covariance= [1e-3, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 1e-3]

            imu_msg.linear_acceleration.x = acceleration[0]
            imu_msg.linear_acceleration.y = acceleration[1]
            imu_msg.linear_acceleration.z = acceleration[2]
            imu_msg.linear_acceleration_covariance= [1e-3, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 1e-3]

            mag_msg.magnetic_field.x = magnetometer[0]
            mag_msg.magnetic_field.y = magnetometer[1]
            mag_msg.magnetic_field.z = magnetometer[2]

            imu_pub.publish(imu_msg)
            mag_pub.publish(mag_msg)

            location_msg.longitude = longitude_imu
            location_msg.latitude = latitude_imu
            location_msg.altitude = altitude_imu
            location_pub.publish(location_msg)

altitude_imu = 0
longitude_imu = 0
latitude_imu = 0
version = 0
key = 0
buff = {}
angularVelocity = [0, 0, 0]
acceleration = [0, 0, 0]
magnetometer = [0, 0, 0]
angle_degree = [0, 0, 0]
global wt_imu, frame_id

if __name__ == "__main__":
    global wt_imu, frame_id
    wt_imu = serial.Serial()
    python_version = platform.python_version()[0]

    find_ttyUSB()
    rospy.init_node("imu_node")
    port = rospy.get_param("~port", "/dev/ttyUSB0")
    baudrate = rospy.get_param("~baud", 9600)
    frame_id = rospy.get_param("~frame_id", "imu_link")
    print("IMU Type: Normal Port:%s baud:%d" %(port,baudrate))
    
    imu_msg = Imu()
    mag_msg = MagneticField()
    location_msg = NavSatFix()

    try:
        wt_imu = serial.Serial(port=port, baudrate=baudrate, timeout=10)
        if wt_imu.isOpen():
            rospy.loginfo("\033[32mSerial port enabled successfully...\033[0m")
        else:
            wt_imu.open()
            rospy.loginfo("\033[32mSerial port enabled successfully...\033[0m")
    except Exception as e:
        print(e)
        rospy.loginfo("\033[31mFailed to open the serial port\033[0m")
        exit(0)
    else:
        imu_pub = rospy.Publisher("imu", Imu, queue_size=10)
        mag_pub = rospy.Publisher("mag", MagneticField, queue_size=10)
        location_pub = rospy.Publisher("location",NavSatFix,queue_size=10)

        while not rospy.is_shutdown():
            try:
                buff_count = wt_imu.inWaiting()
                if buff_count > 0:
                    buff_data = wt_imu.read(buff_count)
                    for i in range(0, buff_count):
                        handleSerialData(buff_data[i])
                else:
                    # 串口空闲时歇 1ms，避免忙等空转 100% CPU
                    time.sleep(0.001)
            except Exception as e:
                print("exception:" + str(e))
                print("imu loss of connection, poor contact, or broken wire")
                exit(0)

