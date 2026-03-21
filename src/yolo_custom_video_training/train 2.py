import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

import logging
from pathlib import Path

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.multiple_video_dataset import MultipleVideoDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqDataset
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector

logging.basicConfig(level=logging.INFO)

print("Works!")

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()

config['model']['path'] = "/workspace/fast_slow_3_finetune.pt"
config['train']['tensorboard_path'] = "/workspace/Artifacts/Logs/"
config['train']['epochs'] = 150
config['data']['test_percent'] = 0.0015
config['train']['batch_size'] = 2
config['lr']['initial_lr'] = 0.001

labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

t0 = time.time()

voc_ds = MultipleVideoDataset("/workspace/MultipleVideo")
voc_ds.build_cache()
voc_ds = YoloSeqDataset(voc_ds, 640)

print(f"Caching time: {time.time() - t0}")

logging.info(f'Cache built')

objectDetector = CustomVideoObjectDetector(config, labels)
objectDetector.load_weights("/workspace/YoloFastSlowContinue.pt", "/workspace/yolo11n_voc_2.pt", "/workspace/yolo11x_voc_2.pt")

freeze = {
    "backbone": (1, 15, 0.0005),
}

objectDetector.train(voc_ds, None, None)
