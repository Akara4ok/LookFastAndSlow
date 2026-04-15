import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.single_video_dataset import SingleVideoDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['train']['batch_size'] = 1
config['model']['img_size'] = 640

objectDetector = CustomVideoObjectDetector(config, VOCDataset.VOC_CLASSES)
objectDetector.load_weights("Model/Yolo/TrueYoloFastAndSlow.pt", "Model/Final/yolo11n.pt", "Model/Final/yolo11x.pt")

# voc_ds = VOCDataset("Data/VOCdevkit", "2007", "train", use_cache=False)
voc_ds = VOCDataset("Data/VOCDevKitTest", "2007", "test", use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)

map = objectDetector.test(voc_ds, 96)
print(map)