#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
from itertools import product
from ais_bench.infer.interface import InferSession

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
    YUNET_OM = os.path.join(current_file_path, "model/yunet_120x160.om")

    def __init__(
        self,
        model_path=YUNET_OM,
        input_shape=[160, 120],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
    ):
        print('--> Load OM model')
        self.session = InferSession(0, model_path)
        # 各種設定
        self.input_shape = input_shape  # [w, h]
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.topk = topk
        self.keep_topk = keep_topk

        # priors生成
        self.priors = None
        self._generate_priors()

    def inference(self, image):
        # 前処理
        temp_image = copy.deepcopy(image)
        temp_image = self._preprocess(temp_image)
        # 推論
        result = self.session.infer([temp_image], mode='static')
        # 後処理
        bboxes, landmarks, scores = self._postprocess(result)
        
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

    def _generate_priors(self):
        w, h = self.input_shape

        feature_map_2th = [
            int(int((h + 1) / 2) / 2),
            int(int((w + 1) / 2) / 2)
        ]
        feature_map_3th = [
            int(feature_map_2th[0] / 2),
            int(feature_map_2th[1] / 2)
        ]
        feature_map_4th = [
            int(feature_map_3th[0] / 2),
            int(feature_map_3th[1] / 2)
        ]
        feature_map_5th = [
            int(feature_map_4th[0] / 2),
            int(feature_map_4th[1] / 2)
        ]
        feature_map_6th = [
            int(feature_map_5th[0] / 2),
            int(feature_map_5th[1] / 2)
        ]

        feature_maps = [
            feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th
        ]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.MIN_SIZES[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.STEPS[k] / w
                    cy = (i + 0.5) * self.STEPS[k] / h

                    priors.append([cx, cy, s_kx, s_ky])

        self.priors = np.array(priors, dtype=np.float32)

    def _preprocess(self, image):
        # BGR -> RGB 変換
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # リサイズ
        image = cv.resize(
            image,
            (self.input_shape[0], self.input_shape[1]),
            interpolation=cv.INTER_LINEAR,
        )

        # リシェイプ
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.input_shape[1], self.input_shape[0])

        return image

    def _postprocess(self, result):
        # 結果デコード
        dets = self._decode(result)

        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.conf_th,
            nms_threshold=self.nms_th,
            top_k=self.topk,
        )

        # bboxes, landmarks, scores へ成形
        scores = []
        bboxes = []
        landmarks = []
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            if len(dets.shape) == 3:
                dets = np.squeeze(dets, axis=1)
            for det in dets[:self.keep_topk]:
                scores.append(det[-1])
                bboxes.append(det[0:4].astype(np.int32))
                landmarks.append(det[4:14].astype(np.int32).reshape((5, 2)))

        return bboxes, landmarks, scores

    def _decode(self, result):
        loc, conf, iou = result

        # スコア取得
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]

        _idx = np.where(iou_scores < 0.)
        iou_scores[_idx] = 0.
        _idx = np.where(iou_scores > 1.)
        iou_scores[_idx] = 1.
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(self.input_shape)

        # バウンディングボックス取得
        bboxes = np.hstack(
            ((self.priors[:, 0:2] +
              loc[:, 0:2] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self.VARIANCE)) *
             scale))
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # ランドマーク取得
        landmarks = np.hstack(
            ((self.priors[:, 0:2] +
              loc[:, 4:6] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 6:8] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 8:10] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 10:12] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 12:14] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale))

        dets = np.hstack((bboxes, landmarks, scores))

        return dets
