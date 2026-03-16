import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path
from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.single_video_dataset import SingleVideoDataset
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector
from ObjectDetector.Yolo.seq_visualizator import SequenceVisualizator

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['img_size'] = 640


objectDetector = CustomVideoObjectDetector(config, VOCDataset.VOC_CLASSES, True)
objectDetector.load_weights("Model/Yolo/fast_slow_3.pt", "Model/Yolo/yolo11n_voc.pt", "Model/Yolo/yolo11x_voc.pt")

voc_ds = VOCDataset("Data/VOCDevKitTest", "2007", "test", use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)

# voc_ds = SingleVideoDataset("Data/SingleVideo")

visualizator = SequenceVisualizator(objectDetector, config)
visualizator.process(voc_ds)
