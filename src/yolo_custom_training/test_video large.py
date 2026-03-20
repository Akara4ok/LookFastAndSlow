import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

from ConfigUtils.config import Config
from ObjectDetector.Yolo.custom_image_object_detector import CustomImageObjectDetector
from ObjectDetector.video_processor import VideoProcessor
from Dataset.voc_dataset import VOCDataset


logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['train']['batch_size'] = 1
config['model']['img_size'] = 640

objectDetector = CustomImageObjectDetector(config, VOCDataset.VOC_CLASSES)
objectDetector.load_weights("Model/Yolo/yolo11x_voc.pt")

videoProcessor = VideoProcessor(objectDetector)
videoProcessor.process_video("Data/video/4.mp4", "Data/output.mp4", True)