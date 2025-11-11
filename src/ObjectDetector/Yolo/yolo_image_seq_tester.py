import logging
from typing import Dict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO

from ObjectDetector.map import MeanAveragePrecision
from ObjectDetector.Yolo.general_video_object_detector import GeneralVideoObjectDetector
from Dataset.Yolo.YoloSegDataset import YoloSeqTestDataset, InferenceTransform

class YoloImageSeqTester(GeneralVideoObjectDetector):
    def __init__(self, config: Dict, labels = None, map_classes = None, device: torch.device | str | None = None):
        super().__init__(config, labels, map_classes, device)
        self.model: nn.Module = None
    
    def load_weights(self, weights_path: str):
        self.model = YOLO(weights_path)
        if self.labels is None:
            self.labels = self.model.names
        self.model.to(self.device)
        logging.info(f"Weights loaded from {weights_path} to device {self.device}")
    
    @torch.no_grad()
    def predict_seq(self, batch: np.ndarray) -> list[dict]:
        B, T, C, H, W = batch.shape
        results = []

        for t in range(T):
            x_t = batch[:, t]
        
            res = self.model.predict(x_t, verbose = False)
            results.append(self.postprocess_single(res))

        return results
    