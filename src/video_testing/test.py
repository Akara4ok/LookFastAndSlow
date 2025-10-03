import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from ConfigUtils.config import Config
from pathlib import Path
import logging
from ObjectDetector.image_object_detector import ImageObjectDetector
from ObjectDetector.video_processor import VideoProcessor
from ObjectDetector.Anchors.mobilenet_anchors import specs

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/vocqwe.weights.h5"
config['data']['path'] = "Data/VOCDevKitTest"
config['train']['batch_size'] = 32
config['anchors']['post_iou_threshold'] = 0.2
config['anchors']['confidence'] = 0.6
config['anchors']['top_k_classes'] = 200

labels = [ "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

torch.set_num_threads(8)

objectDetector = ImageObjectDetector(labels, config, specs)
objectDetector.load_weights(config["model"]["path"])

videoProcessor = VideoProcessor(objectDetector)
videoProcessor.process_video("Data/test.mp4", "Data/output.mp4", True)

