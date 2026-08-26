#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# set_robot_description.py —— 加载小车 URDF 到 /robot_description
# ------------------------------------------------------------
# 2026-08-14: 替代 rosparam set + sed 删注释方案。
# rosparam set 用 yaml 解析, xacro 输出里的中文注释(含 "已移除: ")会炸;
# sed 删注释又会破坏 XML。这里直接用 rospy.set_param (纯字符串, 无解析)。
# 用法: source 环境后执行本脚本。
# ============================================================
import subprocess
import os
import rospkg
import rospy

URDF_XACRO = os.path.join(rospkg.RosPack().get_path('robot_description'), 'urdf', 'mowen.urdf.xacro')
XACRO_BIN = '/opt/ros/melodic/bin/xacro'


def main():
    rospy.init_node('set_robot_description', anonymous=True)
    out = subprocess.check_output(
        [XACRO_BIN, '--inorder', URDF_XACRO],
        stderr=subprocess.DEVNULL).decode('utf-8', 'replace')
    rospy.set_param('/robot_description', out)
    print('robot_description 已加载: %d 字符, %d 行' % (len(out), out.count('\n')))
    # 验证 XML 结构完整
    import xml.etree.ElementTree as ET
    ET.fromstring(out)
    print('XML 结构校验通过')


if __name__ == '__main__':
    main()
