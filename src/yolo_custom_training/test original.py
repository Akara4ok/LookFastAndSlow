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
from ObjectDetector.Yolo.custom_image_object_detector import CustomImageObjectDetector
from visualize import visulize


logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/yolo11x.pt"
config['model']['img_size'] = 640
config['train']['epochs'] = 10
config['data']['path'] = "Data/VOCDevKitTest"
config['train']['batch_size'] = 1

map = {
    0: 14,
    1: 1,
    2: 6,
    3: 13,
    4: 0,
    5: 5,
    6: 18,
    7: 6,
    8: 3,
    14: 2,
    15: 7,
    16: 11,
    17: 12,
    18: 16,
    19: 9,
    39: 4,
    56: 8,
    57: 17,
    58: 15,
    60: 10,
    62: 0
}

labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

objectDetector = CustomImageObjectDetector(labels, config, map)
objectDetector.load_weights("Model/yolo11x.pt")

test_ds = VOCDataset(config['data']['path'], "2007", "test", False)
# test_ds = YoloDataset(train_ds, 640)

map = objectDetector.test(test_ds, 96)
print(map)

# test_ds = VOCDataset(config['data']['path'], "2007", "test", False)
# for img, tgt in test_ds:
#     prediction = objectDetector.predict(img)
#     visulize(img / 255, prediction, labels)





