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

    def process(self, ds: Dataset):
        ds = ImageSeqVideoDataset(ds)
        test_ds = YoloSeqTestDataset(ds, self.config["model"]["img_size"])

        for j, (imgs, tgt) in enumerate(ds):
            n = len(imgs)
            fig, axes = plt.subplots(2, 3, figsize=(10, 6))

            img_batch, _ = test_ds[j]
            predicts = self.objectDetector.predict_seq(torch.unsqueeze(img_batch.to("cuda"), 0))

            for i, ax in enumerate(axes.flat):
                boxes = tgt[i]["boxes"] 
                labels = tgt[i]["labels"]

                for (box, label) in zip(boxes, labels): 
                    xmin, ymin, xmax, ymax = box
                    h = ymax - ymin
                    w = xmax - xmin
                    rect = plt.Rectangle((xmin, ymin), w , h,
                                        fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(xmin, ymin - 5, f"{self.labels[label]}", color='red', fontsize=8)

                for (box, label) in zip(predicts[i]["boxes"], predicts[i]["classes"]):
                    xmin, ymin, xmax, ymax = box
                    
                    img_w = imgs[0].shape[1]
                    img_h = imgs[0].shape[0]

                    xmin *= img_w
                    ymin *= img_h
                    xmax *= img_w
                    ymax *= img_h

                    h = ymax - ymin
                    w = xmax - xmin
                    rect = plt.Rectangle((xmin, ymin), w , h,
                                        fill=False, edgecolor='green', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(xmin, ymin - 5, f"{self.labels[label]}", color='green', fontsize=8)
                
                ax.axis("off")
                ax.set_title("Image " + str(i))

                # ax.imshow(img_batch[i].permute(1, 2, 0).cpu().numpy())
                ax.imshow(imgs[i] / 255)

            plt.show()

    