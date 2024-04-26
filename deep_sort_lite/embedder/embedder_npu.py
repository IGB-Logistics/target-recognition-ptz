from ais_bench.infer.interface import InferSession
import os
import logging

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

current_file_path = os.path.dirname(os.path.realpath(__file__))
MOBILENETV2_OM = os.path.join(current_file_path, "./weights/mobilenetv2.om")
SFACE_OM = os.path.join(current_file_path, "./weights/sface_112x112.om")

NPY_FLOAT32 = 11
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_DEVICE, ACL_HOST = 0, 1
ACL_SUCCESS = 0

MOBILENETV2_INPUT_WIDTH = 224
SFACE_INPUT_WIDTH = 112

def batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx : min(ndx + bs, l)]


class MobileNetv2_NPU_Embedder(object):
    """
    MobileNetv2_NPU_Embedder loads a Mobilenetv2 pretrained on Imagenet1000, with classification layer removed, exposing the bottleneck layer, outputing a feature of size 1280.

    Params
    ------
    - om_model_path (optional, str) : path to mobilenetv2 model weights, defaults to the model file in ./mobilenetv2
    - max_batch_size (optional, int) : max batch size for embedder, defaults to 16
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
    """

    def __init__(
        self, om_model_path=None, init_npu=True, max_batch_size=16, bgr=True
    ):
        if om_model_path is None:
            om_model_path = MOBILENETV2_OM
        assert os.path.exists(
            om_model_path
        ), f"Mobilenetv2 model path {om_model_path} does not exists!"
        
        self.device_id = 0
        self.max_batch_size = max_batch_size
        self.bgr = bgr

        logger.info("MobileNetV2 Embedder for Deep Sort initialised")
        logger.info(f"- max batch size: {self.max_batch_size}")
        logger.info(f"- expects BGR: {self.bgr}")
        
        self.session = InferSession(0, om_model_path)


        zeros = np.zeros((100, 100, 3), dtype=np.uint8)
        self.predict([zeros])  # warmup

    def preprocess(self, np_image):
        """
        Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        Torch Tensor

        """
        if self.bgr:
            np_image_rgb = np_image[..., ::-1]
        else:
            np_image_rgb = np_image

        input_image = cv2.resize(np_image_rgb, (MOBILENETV2_INPUT_WIDTH, MOBILENETV2_INPUT_WIDTH))
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_image = trans(input_image)
        input_image = input_image.view(1, 3, MOBILENETV2_INPUT_WIDTH, MOBILENETV2_INPUT_WIDTH)

        return input_image[0]

    def predict(self, np_images):
        """
        batch inference

        Params
        ------
        np_images : list of ndarray
            list of (H x W x C), bgr or rgb according to self.bgr

        Returns
        ------
        list of features (np.array with dim = 1280)

        """
        all_feats = []

        preproc_imgs = [self.preprocess(img) for img in np_images]
        
        inputs = [torch.unsqueeze(img, dim=0).numpy() for img in preproc_imgs]
        # execute inference, inputs is ndarray list and outputs is ndarray list
        outputs = self.session.infer_pipeline(inputs, mode='static')
        all_feats = [output[0][0] for output in outputs]
        
        return all_feats
    

class Sface_NPU_Embedder(object):
    """
    Sface_NPU_Embedder loads a Sface pretrained, with classification layer removed, exposing the bottleneck layer, outputing a feature of size 128.

    Params
    ------
    - om_model_path (optional, str) : path to mobilenetv2 model weights, defaults to the model file in ./mobilenetv2
    - max_batch_size (optional, int) : max batch size for embedder, defaults to 16
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
    """

    def __init__(
        self, om_model_path=None, init_npu=True, max_batch_size=16, bgr=True
    ):
        if om_model_path is None:
            om_model_path = SFACE_OM
        assert os.path.exists(
            om_model_path
        ), f"sface model path {om_model_path} does not exists!"
        
        self.device_id = 0
        self.max_batch_size = max_batch_size
        self.bgr = bgr

        logger.info("Sface Embedder for Deep Sort initialised")
        logger.info(f"- max batch size: {self.max_batch_size}")
        logger.info(f"- expects BGR: {self.bgr}")
        
        self.session = InferSession(0, om_model_path)


        zeros = np.zeros((112, 112, 3), dtype=np.uint8)
        self.predict([zeros])  # warmup

    def preprocess(self, image):
        assert isinstance(image, np.ndarray), 'image must be numpy array'
        image = cv2.resize(
            image,
            (SFACE_INPUT_WIDTH, SFACE_INPUT_WIDTH),
            interpolation=cv2.INTER_LINEAR,
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

        preproc_imgs = [self.preprocess(img) for img in np_images]

        # execute inference, inputs is ndarray list and outputs is ndarray list
        outputs = self.session.infer_pipeline(preproc_imgs, mode='static')
        all_feats = [output[0][0] for output in outputs]
        
        return all_feats
    