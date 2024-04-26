#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from itertools import product

import cv2 as cv
import numpy as np
import onnxruntime

class SfaceONNX(object):

    def __init__(
        self,
        model_path,
        input_shape=[112,112],
        score_th=0.25,
    ):
        # モデル読み込み
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=so)

        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        # 各種設定
        self.input_shape = input_shape  # [w, h]
        self.score_th = score_th
        
        # self.feature_vectors = None
        # self.face_ids = []

    def inference(self, image):
        # 前処理
        temp_image = copy.deepcopy(image)
        temp_image = self._preprocess(temp_image)

        # 推論
        result = self.onnx_session.run(
            None,
            {self.input_name: temp_image},
        )
        result = np.array(result[0][0])

        # 後処理
        # self._postprocess(result)

        return result

    def _preprocess(self, image):
        image = cv.resize(
            image,
            (self.input_shape[0], self.input_shape[1]),
            interpolation=cv.INTER_LINEAR,
        )
        image = image.transpose(2, 0, 1)
        image = image.astype('float32')
        image = np.expand_dims(image, axis=0)
        
        return image

    def _postprocess(self, result):
        return result

    
    def _cos_similarity(self, X, Y):
        Y = Y.T
        # (128,) x (n, 128) = (n,)
        result = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))
        return result
    


    def _image_rotate(self, image, angle, scale=1.0):
        image_width, image_height = image.shape[1], image.shape[0]
        center = (int(image_width / 2), int(image_height / 2))

        rotation_mat_2d = cv.getRotationMatrix2D(center, angle, scale)

        result_image = cv.warpAffine(
            image,
            rotation_mat_2d,
            (image_width, image_height),
            flags=cv.INTER_CUBIC,
        )

        return result_image