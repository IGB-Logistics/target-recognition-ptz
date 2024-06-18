import os
import cv2 as cv
import numpy as np

class Sface_Embedder(object):
    """
    Sface_NPU_Embedder loads a Sface pretrained, with classification layer removed, exposing the bottleneck layer, outputing a feature of size 128.

    Params
    ------
    - platform (optional, str) : platform to run the model, defaults to 'onnx'
    - max_batch_size (optional, int) : max batch size for embedder, defaults to 16
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
    """

    SFACE_INPUT_WIDTH = 112
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    
    def __init__(
        self, 
        platform='onnx',
        max_batch_size=16, 
        bgr=True
    ):
        
        self.sface = None
        if platform == 'onnx':
            print('Use ONNX Runtime')
            from .sface.sface_onnx import SfaceONNX
            sface_path = os.path.join(self.current_file_path, "./model/sface_112x112.onnx")
            self.sface = SfaceONNX(model_path=sface_path)
        elif platform == 'trt':
            print('Use TRT Model')
            from .sface.sface_trt import SfaceTRT
            sface_path = os.path.join(self.current_file_path, "./model/sface_112x112.trt")
            self.sface = SfaceTRT(model_path=sface_path)
        elif platform == 'om':
            print('Use OM Model')
            from .sface.sface_om import SfaceOM
            sface_path = os.path.join(self.current_file_path, "./model/sface_112x112.om")
            self.sface = SfaceOM(model_path=sface_path)
        elif platform == 'rknn-3566':
            print('Use RKNN Model')
            from .sface.sface_rknn import SfaceRKNN
            sface_path = os.path.join(self.current_file_path, "./model/sface_112x112_for_rk3566.rknn")
            self.sface = SfaceRKNN(model_path=sface_path)
        elif platform == 'rknn-3588':
            print('Use RKNN Model')
            from .sface.sface_rknn import SfaceRKNN
            sface_path = os.path.join(self.current_file_path, "./model/sface_112x112_for_rk3588.rknn")
            self.sface = SfaceRKNN(model_path=sface_path)
        
        self.device_id = 0
        self.max_batch_size = max_batch_size
        self.bgr = bgr


        zeros = np.zeros((112, 112, 3), dtype=np.uint8)
        self.predict([zeros])  # warmup

    def preprocess(self, image):
        assert isinstance(image, np.ndarray), 'image must be numpy array'
        image = cv.resize(
            image,
            (self.SFACE_INPUT_WIDTH, self.SFACE_INPUT_WIDTH),
            interpolation=cv.INTER_LINEAR,
        )
        image = image.transpose(2, 0, 1)
        image = image.astype('float32')
        image = np.expand_dims(image, axis=0)
        
        return image

    def predict(self, np_images):
        """
        batch inference

        Params
        ------
        np_images : list of ndarray
            list of (H x W x C), bgr or rgb according to self.bgr

        Returns
        ------
        list of features (np.array with dim = 128)

        """
        all_feats = []
        
        for img in np_images:
            features = self.sface.inference(img)
            all_feats.append(features)
            
        return all_feats
    