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
from ObjectDetector.Yolo.image_object_detector import ImageObjectDetector
from visualize import visulize


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

objectDetector = ImageObjectDetector(labels, config)
objectDetector.load_weights("Model/vocyolo11x.weights.h5", "Model/yolo11x.pt")

test_ds = VOCDataset(config['data']['path'], "2007", "test", False)
# test_ds = YoloDataset(train_ds, 640)

map = objectDetector.test(test_ds, 96)
print(map)

# test_ds = VOCDataset(config['data']['path'], "2007", "trainval", False)
# for img, tgt in test_ds:
    # prediction = objectDetector.predict(img)
    # visulize(img / 255, prediction, labels)





