import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path
from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from ObjectDetector.Yolo.yolo_image_seq_tester import YoloImageSeqTester
from ObjectDetector.Yolo.seq_visualizator import SequenceVisualizator

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['img_size'] = 640


objectDetector = YoloImageSeqTester(config, VOCDataset.VOC_CLASSES)
objectDetector.load_weights("Model/Yolo/yolo11x_voc.pt")

voc_ds = VOCDataset("Data/VOCdevkit", "2007", "train", use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)

visualizator = SequenceVisualizator(objectDetector, config)
visualizator.process(voc_ds)



