#! /home/adminpc/anaconda3/bin/python
# 在 ros 中运行此文件
import sys

sys.path.append('/home/adminpc/yolov5-master')  # 在ros中，需要将环境添加进取
import rospy
from sit_service_vision.msg import detection
import string
from my_detect import my_detect
from my_deepsort import MyDeepSort


def publi(strs):
    det = detection()
    ID, id, *cls, dist, conf = strs.split(" ")
    det.ID = ID
    det.id = int(id)
    det.cls = "".join(cls)
    det.dist =float(dist[:len(dist)-1])
    det.conf = float(conf)
    pub.publish(det)


if __name__ == '__main__':

    det = my_detect()
    deepsort = MyDeepSort()

    rospy.init_node("yolo_detector")    # 初始化ros节点
    pub = rospy.Publisher("yolo_detection", detection, queue_size=10)  # 定义话题发布函数

    rate = rospy.Rate(1)

    det.action(publi, deepsort.update)   # 将发布函数作为回调函数传入，当有物品检测时，发布检测信息
    rate.sleep()

