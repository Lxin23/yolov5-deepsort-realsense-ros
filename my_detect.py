import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots
from utils.my_dataloaders import rs_load
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import numpy as np
import random


def get_mid_pos(frame, box, depth_data, randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]  # 确定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))  # 确定深度搜索范围
    # print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val // 4, min_val // 4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255, 0, 0), -1)  # 在图像上画个圆

        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波
    # print(distance_list, np.mean(distance_list))
    return np.mean(distance_list)


class Dector:
    def __init__(self):
        self.weights = '/home/adminpc/yolov5-master/yolov5s.pt'  # model path or triton URL
        self.source = '4'  # file/dir/URL/glob/screen/0(webcam)
        self.data = ROOT / 'data/coco128.yaml',  # dataset.yaml path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.1  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = 0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = False  # show results
        self.save_txt = False  # save results to *.txt
        self.save_conf = False  # save confidences in --save-txt labels
        self.save_crop = False  # save cropped prediction boxes
        self.nosave = False  # do not save images/videos
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models
        self.project = ROOT / 'runs/detect'  # save results to project/name
        self.name = 'my_exp'  # save results to project/name
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.vid_stride = 1  # video frame-rate stride


class my_detect:
    def __init__(self):
        self.det = Dector()
        self.init_source_dir()
        self.load_model()
        print("init finish")

    def init_source_dir(self):
        self.source = str(self.det.source)
        self.save_img = not self.det.nosave and not self.source.endswith('.txt')  # save inference images
        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (self.is_url and not self.is_file)
        self.screenshot = self.source.lower().startswith('screen')
        if self.is_url and self.is_file:
            self.source = check_file(self.source)  # download

        # Directories
        self.save_dir = increment_path(Path(self.det.project) / self.det.name,
                                       exist_ok=self.det.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.det.save_txt else self.save_dir).mkdir(parents=True,
                                                                                 exist_ok=True)  # make dir

    def load_model(self):
        self.device = select_device(self.det.device)
        self.model = DetectMultiBackend(self.det.weights, device=self.device, dnn=self.det.dnn, data=self.det.data,
                                        fp16=self.det.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.det.imgsz, s=self.stride)  # check image size
        print("load model finish")

    def action(self, callback, update):
        print("ready")
        # Dataloader
        bs = 1  # batch_size

        # if self.webcam:    
        view_img = check_imshow(warn=True)
        dataset = rs_load(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt,
                          vid_stride=self.det.vid_stride)
        bs = len(dataset)
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup 模型预热
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s, depth_image in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.det.visualize else False
                pred = self.model(im, augment=self.det.augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.det.conf_thres, self.det.iou_thres, self.det.classes,
                                           self.det.agnostic_nms, max_det=self.det.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg

                self.txt_path = str(self.save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.det.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.det.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
                    i = 0
                    outputs = update(xywhs, confs, clss, im0)

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for (*xyxy, conf, cls), output in zip(reversed(det), outputs):
                        distance = get_mid_pos(im0, xyxy, depth_image, 24)

                        if self.det.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.det.save_conf else (cls, *xywh)  # label format
                            with open(f'{self.txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.det.save_crop or self.det.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.det.hide_labels else (self.names[c] if self.det.hide_conf else f'ID-{output[4]} {self.names[c]}  {str(distance / 1000)[:4]}m {conf:.2f}')


                            callback(label)

                            # callback(label)  # 调用回调函数处处结果或将结果通过话题发布

                            annotator.box_label(xyxy, label, color=colors(c, True))  # 将检测结果显示(指检测框和文字)
                        if self.det.save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg',
                                         BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)  # 显示图像
                    cv2.waitKey(1)  # 1 millisecond

                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.det.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.det.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        if self.det.update:
            strip_optimizer(self.det.weights[0])  # update model (to fix SourceChangeWarning)

