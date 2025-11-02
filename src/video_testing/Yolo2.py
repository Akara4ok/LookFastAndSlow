import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ConfigUtils.config import Config
from pathlib import Path
import logging
from ObjectDetector.Yolo.general_image_object_detector import GeneralImageObjectDetector
from ObjectDetector.video_processor import VideoProcessor
from ObjectDetector.SSDLite.Anchors.mobilenet_anchors import specs

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/best.pt"
config['data']['path'] = "Data/VOCDevKitTest"
config['train']['batch_size'] = 32
config['anchors']['post_iou_threshold'] = 0.2
config['anchors']['confidence'] = 0.5
config['anchors']['top_k_classes'] = 200

labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
objectDetector = GeneralImageObjectDetector(config)
objectDetector.load_weights("Model/best.pt")

videoProcessor = VideoProcessor(objectDetector)
videoProcessor.process_video("Data/test.mp4", "Data/output.mp4", True)

