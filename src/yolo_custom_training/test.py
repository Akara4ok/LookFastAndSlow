import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from ObjectDetector.Yolo.custom_image_object_detector import CustomImageObjectDetector

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['img_size'] = 640
config['data']['path'] = "Data/VOCDevKitTest"
config['train']['batch_size'] = 1

objectDetector = CustomImageObjectDetector(config, VOCDataset.VOC_CLASSES)
objectDetector.load_weights("Model/Yolo/yolo11n_voc_2.pt")

test_ds = VOCDataset(config['data']['path'], "2007", "test", False)

map = objectDetector.test(test_ds, 96)
print(map)
