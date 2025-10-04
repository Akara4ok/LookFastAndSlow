import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ConfigUtils.config import Config
from pathlib import Path
import logging
import numpy as np
from Dataset.xml_star_dataset import XMLStarDataset
from Dataset.train_dataset import TrainDataset
from Dataset.test_dataset import TestDataset
from Dataset.voc_dataset import VOCDataset
from ObjectDetector.Yolo.video_object_detector import VideoObjectDetector
from ObjectDetector.SSDLite.image_object_detector import ImageObjectDetector
from ObjectDetector.SSDLite.Anchors.mobilenet_anchors import specs
from visualize import visulize          # unchanged helper

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
# config['model']['path'] = "Model/voc.weights.h5"
config['model']['path'] = "Model/voc_video.pt"
config['data']['path'] = "Data/VOCDevKitTest"
config['train']['batch_size'] = 32
config['anchors']['post_iou_threshold'] = 0.45
config['anchors']['confidence'] = 0.45
config['anchors']['top_k_classes'] = 200

test_ds = VOCDataset("Data/VOCDevKitTest", "2007", "test", 300, False)
# test_ds = TestDataset(VOCDataset("Data/VOCDevKitTest", "2007", "test", 300, False), 300)

img_to_test = 96

labels = [ "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
objectDetector = (labels, config, specs)
# objectDetector = ImageObjectDetector(labels, config, specs)
objectDetector = VideoObjectDetector(labels, config, specs)
objectDetector.load_weights(config["model"]["path"])

# for img, tgt in test_ds:
    # hwc = img.permute(1, 2, 0).clamp(0, 1).numpy() * 255
    # hwc = hwc.astype(np.uint8)
    # prediction = objectDetector.predict(hwc)
    # visulize(hwc / 255, prediction, labels)
    # visulize(img.permute(1, 2, 0), prediction, labels)

map = objectDetector.test(test_ds, img_to_test)
print(map)
