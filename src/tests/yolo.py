import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path
from PIL import Image
import numpy as np

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from ObjectDetector.Yolo.image_object_detector import ImageObjectDetector
from visualize import visulize


logging.basicConfig(level=logging.INFO)

cfg = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
cfg['model']['path'] = "Model/yolo11x.pt"

labels = ["person"] * 80

objectDetector = ImageObjectDetector(None, cfg)
objectDetector.load_weights(cfg["model"]["path"])

test_ds = VOCDataset("Data/VOCDevKitTest", "2007", "test", False)

for img, tgt in test_ds:
    prediction = objectDetector.predict(img)
    visulize(img / 255, prediction, labels)

# prediction = objectDetector.predict(img)


# visulize(img, prediction, labels)

