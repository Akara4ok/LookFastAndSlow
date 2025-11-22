import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.Yolo.YoloDataset import YoloDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqDataset, YoloSeqTestDataset
from ObjectDetector.Yolo.yolo_image_seq_tester import YoloImageSeqTester
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector
from ObjectDetector.Yolo.seq_visualizator import SequenceVisualizator

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/yolo11x.pt"
config['model']['img_size'] = 640
config['train']['epochs'] = 10
config['data']['path'] = "Data/VOCDevKitTest"
config['train']['batch_size'] = 1


objectDetector = CustomVideoObjectDetector(config, VOCDataset.VOC_CLASSES, True)
objectDetector.load_weights("Model/Yolo/fast_slow_2.pt", "Model/Yolo/yolo11n_voc.pt", "Model/Yolo/yolo11x_voc.pt", True)

voc_ds = VOCDataset("Data/VOCDevKitTest", "2007", "test", use_cache=False)

visualizator = SequenceVisualizator(objectDetector, config)
visualizator.process(voc_ds)
