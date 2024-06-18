#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
from itertools import product

import cv2 as cv
import numpy as np
import math

def crop_face_images(image, input_shape, bboxes, landmarks, scores, score_th):
    image_height, image_width = image.shape[0], image.shape[1]

    face_image_list = []
    for score, bbox, landmark in zip(scores, bboxes, landmarks):
        if score_th > score:
            continue
        x1 = int(image_width * (bbox[0] / input_shape[0]))
        y1 = int(image_height * (bbox[1] / input_shape[1]))
        x2 = int(image_width * (bbox[2] / input_shape[0])) + x1
        y2 = int(image_height * (bbox[3] / input_shape[1])) + y1
    
        face_image = copy.deepcopy(image[y1:y2, x1:x2])

        right_eye = landmark[0]
        left_eye = landmark[1]
        mouth = landmark[2]

        a = np.array([((right_eye[0] + left_eye[0]) / 2),
                    ((right_eye[1] + left_eye[1]) / 2)])
        b = np.array([mouth[0], mouth[1]])
        vec = b - a
        angle = math.degrees(np.arctan2(vec[0], vec[1]))

        # face_image = self._image_rotate(face_image, -angle)
        face_image_list.append(face_image)

    return face_image_list

class YuNetDet(object):

    YUNET_SIZE_W = 160
    YUNET_SIZE_H = 120
    MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    STEPS = [8, 16, 32, 64]
    VARIANCE = [0.1, 0.2]
    current_file_path = os.path.dirname(os.path.realpath(__file__))

    def __init__(
        self,
        input_shape=[160, 120],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
        platform='onnx'
    ):
        self.yunet = None
        if platform == 'onnx':
            print('Use ONNX Runtime')
            from .yunet.yunet_onnx import YuNetONNX
            yunet_path = os.path.join(self.current_file_path, "model/yunet_120x160.onnx")
            self.yunet = YuNetONNX(model_path=yunet_path)
        elif platform == 'trt':
            print('Use TRT Model')
            from .yunet.yunet_trt import YuNetTRT
            yunet_path = os.path.join(self.current_file_path, "/model/yunet_120x160.trt")
            self.yunet = YuNetTRT(model_path=yunet_path)
        elif platform == 'om':
            print('Use OM Model')
            from .yunet.yunet_om import YuNetOM
            yunet_path = os.path.join(self.current_file_path, "model/yunet_120x160.om")
            self.yunet = YuNetOM(model_path=yunet_path)
        elif platform == 'rknn-3566':
            print('Use RKNN Model')
            from .yunet.yunet_rknn import YuNetRKNN
            yunet_path = os.path.join(self.current_file_path, "model/yunet_120x160_for_rk3566.rknn")
            self.yunet = YuNetRKNN(model_path=yunet_path)
        elif platform == 'rknn-3588':
            print('Use RKNN Model')
            from .yunet.yunet_rknn import YuNetRKNN
            yunet_path = os.path.join(self.current_file_path, "model/yunet_120x160_for_rk3588.rknn")
            self.yunet = YuNetRKNN(model_path=yunet_path)
    

    def inference(self, image):
        # 前処理
        temp_image = copy.deepcopy(image)
        # 推論
        bboxes, landmarks, scores = self.yunet.inference(temp_image)
        
        # DeepSort
        # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        boxes = bboxes
        confidence = scores
        detection_class = np.ones(len(boxes))
        # boxes_xywh = [[box[0], box[1], box[2], box[3]] for box in boxes]
        boxes_xywh = [[image.shape[1] * box[0] / self.YUNET_SIZE_W, 
                       image.shape[0] * box[1] / self.YUNET_SIZE_H, 
                       image.shape[1] * box[2] / self.YUNET_SIZE_W,
                       image.shape[0] * box[3] / self.YUNET_SIZE_H] for box in boxes]
        bbs = list(zip(boxes_xywh, confidence, detection_class))
        
        return bbs
