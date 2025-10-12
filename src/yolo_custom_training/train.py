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
config['model']['path'] = "Model/yolo11x.pt"
config['train']['epochs'] = 10
config['data']['path'] = "Data/VOCdevkit"
config['train']['batch_size'] = 1


labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

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

objectDetector = CustomImageObjectDetector(labels, config, map)
objectDetector.load_weights("Model/yolo11x.pt")
# objectDetector.load_weights("Model/test.weights.h5", "Model/yolo11x.pt")
# objectDetector.save_checkpoint(objectDetector.model.model, labels, "Model/test.weights.h5")

train_ds = VOCDataset(config['data']['path'], "2007", "trainval", False)
train_ds = YoloDataset(train_ds, 640)

# # objectDetector.train(train_ds)

test_ds = VOCDataset(config['data']['path'], "2007", "trainval", False)
for img, tgt in test_ds:
    prediction = objectDetector.predict(img)
    visulize(img / 255, prediction, labels)
# # prediction = objectDetector.predict(img)
# # visulize(img, prediction, labels)labels

