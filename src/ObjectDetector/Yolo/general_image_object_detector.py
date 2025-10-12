import math
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import ultralytics
from ultralytics import YOLO

from ObjectDetector.Yolo.image_object_detector_base import ImageObjectDetectorBase

class GeneralImageObjectDetector(ImageObjectDetectorBase):
    def __init__(self, config: Dict, map_classes = None, device: torch.device | str | None = None):
        super().__init__(config, map_classes, device)

    def load_weights(self, weights_path: str):
        self.model = YOLO(weights_path)
    
    def train(self, data_yaml_path: str):
        logging.info("Training started")
        self.model.train(
            data=data_yaml_path,
            epochs=self.config["train"]["epochs"],
            imgsz=self.config["model"]["img_size"],
            batch=self.config["train"]["batch_size"],        # підлаштуйте під вашу VRAM
            workers=4,
            device=0,        # 0 = одна GPU, 'cpu' = без GPU
            patience=20,     # рання зупинка
            optimizer="auto",
            lr0=self.config["lr"]["initial_lr"],
            weight_decay=0.0005,
            project="runs_yolo20",
            name="exp_yolo20",
            verbose=True,
            pretrained=True,
            exist_ok=True
        )
