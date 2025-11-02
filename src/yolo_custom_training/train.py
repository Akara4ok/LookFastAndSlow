import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.Yolo.YoloDataset import YoloDataset
from ObjectDetector.Yolo.custom_image_object_detector import CustomImageObjectDetector
from visualize import visulize


logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/yolo11x_custom.pt"
config['train']['epochs'] = 10
config['data']['path'] = "Data/VOCdevkit"
config['train']['batch_size'] = 1


labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

objectDetector = CustomImageObjectDetector(config, labels)
objectDetector.load_weights("Model/yolo11x.pt")

train_ds = VOCDataset(config['data']['path'], "2007", "trainval", False)
train_ds = YoloDataset(train_ds, 640)

objectDetector.train(train_ds)
