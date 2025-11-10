import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

from ConfigUtils.config import Config
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector
from ObjectDetector.video_processor import VideoProcessor


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

objectDetector = CustomVideoObjectDetector(config, labels, True)
objectDetector.load_weights("Model/yolo11x_custom.pt")

videoProcessor = VideoProcessor(objectDetector)
videoProcessor.process_video("Data/test.mp4", "Data/output.mp4", True)