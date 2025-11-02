import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.Yolo.YoloDataset import YoloDataset
from Dataset.Yolo.YoloDataset import YoloDataset
from ObjectDetector.Yolo.general_image_object_detector import GeneralImageObjectDetector
from ObjectDetector.Yolo.Models.custom_nc_model import CustomDetect
from visualize import visulize


logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/yolo11x.pt"
config['model']['img_size'] = 640
config['train']['epochs'] = 10
config['data']['path'] = "Data/VOCdevkit"
config['train']['batch_size'] = 1

objectDetector = GeneralImageObjectDetector(config)
objectDetector.load_weights("Model/yolo11x_test.pt")
objectDetector.train("Data/YoloVoc/data.yaml")


