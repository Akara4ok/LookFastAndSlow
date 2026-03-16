import logging
from typing import Dict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

from ultralytics import YOLO

import matplotlib.pyplot as plt

from ObjectDetector.Yolo.general_video_object_detector import GeneralVideoObjectDetector
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqTestDataset
from ObjectDetector.map import MeanAveragePrecision

class SequenceVisualizator():
    def __init__(self, objectDetector: GeneralVideoObjectDetector, config: Dict, device: torch.device | str | None = None):
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.objectDetector = objectDetector
        self.labels = self.objectDetector.labels

    def scale_box(self, box: tuple, img: np.ndarray) -> tuple:
        xmin, ymin, xmax, ymax = box                    
        img_w = img.shape[2]
        img_h = img.shape[1]

        xmin *= img_w
        ymin *= img_h
        xmax *= img_w
        ymax *= img_h
        
        return xmin, ymin, xmax, ymax

    def visualize_box(self, ax, box: tuple, label: int, color: str) -> None:
        xmin, ymin, xmax, ymax = box
        h = ymax - ymin
        w = xmax - xmin
        rect = plt.Rectangle((xmin, ymin), w , h,
                            fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{self.labels[label]}", color=color, fontsize=8)

    def calculate_map(self, predicts: list, true_tgts: list) -> float:
        metric = MeanAveragePrecision(num_classes=len(self.labels), device=self.device)
        for i, (preds, raw) in enumerate(zip(predicts, true_tgts)):
            tensor_dict = {k: torch.from_numpy(v) for k, v in preds.items()}
            metric.update([tensor_dict], [raw])

        res = metric.compute()
        
        return  res["weighted_mAP"]

    def process(self, ds: Dataset):
        test_ds = YoloSeqTestDataset(ds, self.config["model"]["img_size"])

        for j, (img_batch, true_tgts) in enumerate(test_ds):
            fig, axes = plt.subplots(2, 3, figsize=(10, 6))

            predicts = self.objectDetector.predict_seq(torch.unsqueeze(img_batch.to("cuda"), 0))

            for i, ax in enumerate(axes.flat):
                boxes = true_tgts[i]["boxes"].cpu().numpy()
                labels = true_tgts[i]["labels"].cpu().numpy()

                for (box, label) in zip(boxes, labels):
                    self.visualize_box(ax, self.scale_box(box, img_batch[0]), label, 'red')

                # for (box, label) in zip(predicts[i]["boxes"], predicts[i]["classes"]):
                    # self.visualize_box(ax, self.scale_box(box, img_batch[0]), label, 'green')
                
                ax.axis("off")
                ax.set_title("Image " + str(i))

                ax.imshow(img_batch[i].permute(1, 2, 0).cpu().numpy())
                # ax.imshow(imgs[i] / 255)

            print("Weighted map on image:", self.calculate_map(predicts, true_tgts))

            plt.show()

    