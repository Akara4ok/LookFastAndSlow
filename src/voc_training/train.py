import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ConfigUtils.config import Config
from pathlib import Path
import logging
from Dataset.SSDLite.xml_star_dataset import XMLStarDataset
from Dataset.voc_dataset import VOCDataset
from ObjectDetector.SSDLite.image_object_detector import ImageObjectDetector
from ObjectDetector.SSDLite.Anchors.mobilenet_anchors import specs

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/vocqwe.weights.h5"
config['train']['epochs'] = 100
config['data']['path'] = "Data/VOCDevKit"
config['anchors']['iou_threshold'] = 0.45
config['anchors']['post_iou_threshold'] = 0.2
config['anchors']['confidence'] = 0.5
config['anchors']['top_k_classes'] = 200
config['train']['batch_size'] = 4

train_ds = VOCDataset("Data/VOCdevkit", "2007", "trainval")

objectDetector = ImageObjectDetector([ "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
], config, specs)
objectDetector.train(train_ds)
