import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqDataset
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector
from visualize import visulize


logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/yolo11x_custom.pt"
config['train']['epochs'] = 10
config['data']['path'] = "Data/VOCdevkit"
config['train']['batch_size'] = 1
config['model']['img_size'] = 640
config["data"]["test_percent"] = 0.01


labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

objectDetector = CustomVideoObjectDetector(config, labels)
objectDetector.load_weights("Model/yolo11x_custom.pt")

voc_ds = VOCDataset("Data/VOCdevkit", "2007", "trainval", use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)

map = objectDetector.test(voc_ds, 5)
print(map)