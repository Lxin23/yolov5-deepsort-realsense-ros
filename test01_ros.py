#! /usr/bin/env/python
# 在 ros 中运行此文件
import sys

sys.path.append('/home/adminpc/yolov5-master')  # 在ros中，需要将环境添加进取

import rospy
from std_msgs.msg import String  # 这个和上面个都是ros的功能包，在ros下才能使用
from my_detect import my_detect
from my_deepsort import MyDeepSort

if __name__ == '__main__':

    det = my_detect()
    deepsort = MyDeepSort()

    rospy.init_node("yolo_detector")    # 初始化ros节点
    pub = rospy.Publisher("yolo_detection", String, queue_size=10)  # 定义话题发布函数

    rate = rospy.Rate(1)

    det.action(pub.publish, deepsort.update)   # 将发布函数作为回调函数传入，当有物品检测时，发布检测信息
    rate.sleep()
