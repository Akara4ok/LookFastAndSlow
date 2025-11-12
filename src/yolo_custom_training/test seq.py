import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path
from PIL import Image
import numpy as np

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.Yolo.YoloDataset import YoloDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqDataset
from ObjectDetector.Yolo.yolo_image_seq_tester import YoloImageSeqTester


logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/yolo11x.pt"
config['model']['img_size'] = 640
config['train']['epochs'] = 10
config['data']['path'] = "Data/VOCDevKitTest"
config['train']['batch_size'] = 1


labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# objectDetector = GeneralImageObjectDetector(config, labels)
objectDetector = YoloImageSeqTester(config, labels)
objectDetector.load_weights("Model/Yolo/yolo11x_voc.pt")

voc_ds = VOCDataset("Data/VOCdevkit", "2007", "trainval", use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)

map = objectDetector.test(voc_ds, 96)
print(map)



