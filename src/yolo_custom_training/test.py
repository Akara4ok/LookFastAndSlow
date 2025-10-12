import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path
from PIL import Image
import numpy as np

from Dataset.voc_dataset import VOCDataset
from Dataset.Yolo.CustomImageToYoloSaver import CustomImageToYoloSaver
from torch.utils.data import random_split

logging.basicConfig(level=logging.INFO)

labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

ds = VOCDataset("Data/VOCdevkit", "2007", "trainval", False)

test_ratio = 0.2
test_len = int(len(ds) * test_ratio)
train_len = len(ds) - test_len

train_ds, val_ds = random_split(ds, [train_len, test_len])
CustomImageToYoloSaver().save(labels, "Data/YoloVoc", train_ds, val_ds)