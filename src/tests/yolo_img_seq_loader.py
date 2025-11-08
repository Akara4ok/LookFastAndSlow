import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqDataset, YoloSeqTestDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector

import matplotlib.pyplot as plt

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['train']['batch_size'] = 8

voc_ds = VOCDataset("Data/VOCdevkit", "2007", "trainval", use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)
voc_ds = YoloSeqDataset(voc_ds, 640)

labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

detector = CustomVideoObjectDetector(config, labels)

loader = detector._make_loader(voc_ds, False)

for batch in loader:
    print("Batch images shape:", batch["images"].shape) # [8, 6, 3, 640, 640]

    print("Batch boxes len:", len(batch["boxes"])) # 8
    print("Seq boxes len:", len(batch["boxes"][0])) # 6
    print("Frame boxes shape:", batch["boxes"][0][0].shape) # [3, 4]

    print("Batch labels len:", len(batch["labels"])) # 8
    print("Seq labels len:", len(batch["labels"][0])) # 6
    print("Frame labels shape:", batch["labels"][0][0].shape) # 3
    
    print("="*125)
    
    ultra_batch = detector._prep_batch_for_ultra_video(batch)

    print("Batch images shape:", ultra_batch["imgs"].shape) # [8, 6, 3, 640, 640]

    gt = ultra_batch["gt"]
    print("Sequens len:", len(gt)) # 6
    print("Batch frame shape:", gt[0]["imgs"].shape) # [8, 3, 640, 640]
    print("Batch boxes shape:", gt[0]["bboxes"].shape) # [15, 4]
    print("Batch labels shape:", gt[0]["cls"].shape) # [15, 1]
    print("Batch batch_idx shape:", gt[0]["batch_idx"].shape) # 15

    break




