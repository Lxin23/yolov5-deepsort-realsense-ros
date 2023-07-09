import pyrealsense2 as rs
import torch
import numpy as np
import cv2
import os
import math
from pathlib import Path
from utils.general import clean_str, check_requirements, is_colab, is_kaggle, LOGGER
from urllib.parse import urlparse
from threading import Thread
from utils.augmentations import letterbox

class rs_load:
    def __init__(self,source, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.transforms = transforms
        self.vid_stride = vid_stride
        self.source = source
    
        self.pipeline = rs.pipeline() 
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        # 初始化参数配置
        if cv2.waitKey(1) == ord('q'):
            self.pipeline.stop()
            cv2.destroyAllWindows()
            raise StopIteration

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame:
            LOGGER.info('realsense error')
        else:
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            im0 = color_image.copy()
            im = np.stack([letterbox(x, self.img_size, self.stride, self.auto)[0] for x in im0])  # resize
            im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)
            opencv_img = np.array([cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)])

            im = opencv_img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

            path = [clean_str(x) for x in self.source]

        return path, im, opencv_img, None, '', depth_image
    
    def __len__(self):
        return len(self.source)  # 1E12 frames = 32 streams at 30 FPS for 30 years
