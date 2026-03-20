import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

from ConfigUtils.config import Config
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector
from ObjectDetector.video_processor import VideoProcessor
from Dataset.voc_dataset import VOCDataset

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['train']['batch_size'] = 1
config['model']['img_size'] = 640

objectDetector = CustomVideoObjectDetector(config, VOCDataset.VOC_CLASSES, True)
objectDetector.load_weights("Model/Final/YoloFastAndSlow.pt", "Model/Final/yolo11n_voc.pt", "Model/Final/yolo11x_voc.pt")
objectDetector.set_nms_params(0.45, 0.4)

videoProcessor = VideoProcessor(objectDetector)
videoProcessor.process_video("Data/test.mp4", "Data/output.mp4", True)