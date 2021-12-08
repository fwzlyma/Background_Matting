# -*- encoding: utf-8 -*-
# @Time: 2021/11/18 21:50
# @Author: fwzlym
# @File: model_sys.py
# @Software: PyCharm
# @Introduce: preprocess: load model and process image for its masksDL
# from __future__ import print_function
import test_segmentation_deeplab as mmask
import tensorflow as tf
from six.moves import urllib

from PIL import Image
import os, glob
from functions import *

class model:
    def __init__(self):
        self.dir_name = "sample_data/input"
        self.model_dir = 'deeplab_model'
        LABEL_NAMES = np.asarray([
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
        ])

        FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
        FULL_COLOR_MAP = mmask.label_to_color_image(FULL_LABEL_MAP)

        self.MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

        self._DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        self._MODEL_URLS = {
            'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
            'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
            'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
            'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }
        _TARBALL_NAME = self._MODEL_URLS[self.MODEL_NAME]
        self.download_path = os.path.join(self.model_dir, _TARBALL_NAME)

    def load_model(self):
        # m_mask = mmask.DeepLabModel()
        if not os.path.exists(self.model_dir):
            tf.compat.v1.gfile.MakeDirs(self.model_dir)
        if not os.path.exists(self.download_path):
            print('downloading model to %s, this might take a while...' % self.download_path)
            urllib.request.urlretrieve(self._DOWNLOAD_URL_PREFIX + self._MODEL_URLS[self.MODEL_NAME],
                                       self.download_path)
            print('download completed! loading DeepLab model...')
        self.MODEL = mmask.DeepLabModel(self.download_path)
        print('model loaded successfully!')
    def produce_maskDL(self):
        list_im = glob.glob(self.dir_name + '/*_img.png')
        list_im.sort()
        for i in range(0, len(list_im)):
            image = Image.open(list_im[i])
            res_im, seg = self.MODEL.run(image)
            seg = cv2.resize(seg.astype(np.uint8), image.size)
            mask_sel = (seg == 15).astype(np.float32)
            name = list_im[i].replace('img', 'masksDL')
            cv2.imwrite(name, (255 * mask_sel).astype(np.uint8))
        str_msg = 'finish: ' + self.dir_name
        print(str_msg)

if __name__ == '__main__':
    a = model()
    a.load_model()
    a.produce_maskDL()