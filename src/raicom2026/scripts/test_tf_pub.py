#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# test_tf_pub.py —— 测试环境静态 TF 发布 (2026-08-14)
# ------------------------------------------------------------
# 替代 tf2_ros/tf 的 static_transform_publisher (roslaunch 定位
# node 失败 / 后台进程随 shell 被杀 / 同名冲突, 一堆坑)。
# 直接以 latched publisher 发布 /tf_static (标准 tf2_msgs/TFMessage),
# 无任何 tf2_ros 依赖, python3 可用。
#
# 发布的变换 (测试环境恒等链; 比赛由 AMCL/EKF/bringup 提供真实值):
#   map -> odom              (0,0,0)
#   odom -> base_footprint   (0,0,0)
#   base_link -> camera_link (0.2, 0, 0.2)   # 相机实测前0.2m 高0.2m
# ============================================================
import rospy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


def make_tf(parent, child, x, y, z):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    t.transform.rotation.w = 1.0
    return t


def main():
    rospy.init_node('test_tf_pub', anonymous=False)
    # 2026-08-15 参数化:
    #   ~publish_map_chain=true  -> 测试环境: map->odom->base_footprint 恒等链
    #   ~publish_map_chain=false -> 比赛环境: 只发 base_link->camera_link
    #       (map->odom 由 AMCL 提供, odom->base_footprint 由 EKF 提供)
    pub_map = rospy.get_param('~publish_map_chain', False)
    cam_x = rospy.get_param('~cam_x', 0.2)
    cam_y = rospy.get_param('~cam_y', 0.0)
    cam_z = rospy.get_param('~cam_z', 0.2)
    pub = rospy.Publisher('/tf_static', TFMessage, queue_size=10, latch=True)
    msg = TFMessage()
    msg.transforms = []
    if pub_map:
        msg.transforms.append(make_tf('map', 'odom', 0.0, 0.0, 0.0))
        msg.transforms.append(make_tf('odom', 'base_footprint', 0.0, 0.0, 0.0))
    msg.transforms.append(make_tf('base_link', 'camera_link', cam_x, cam_y, cam_z))
    rospy.sleep(0.5)
    pub.publish(msg)
    rospy.loginfo('[test_tf] 已发布 %d 个静态变换 (map链=%s, camera=(%.2f,%.2f,%.2f))',
                  len(msg.transforms), pub_map, cam_x, cam_y, cam_z)
    rospy.spin()


if __name__ == '__main__':
    main()
