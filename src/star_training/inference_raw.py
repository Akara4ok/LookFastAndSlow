import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from ConfigUtils.config import Config
from ObjectDetector.SSDLite.image_object_detector import ImageObjectDetector
from visualize import visulize          # unchanged helper
from ObjectDetector.SSDLite.Anchors.mobilenet_anchors import specs

logging.basicConfig(level=logging.INFO)

cfg = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
cfg['model']['path'] = "Model/star.weights.h5"
cfg['anchors']['confidence'] = 0.01
cfg['anchors']['top_k_classes'] = 200

labels = ["None", "Star"]                       # background idx 0 + 1 class
detector = ImageObjectDetector(labels, cfg, specs)

detector.load_weights(cfg["model"]["path"])

img_path = Path("Data/images/a (6).jpg")
img      = Image.open(img_path).convert("RGB")           # PIL Image
img      = TF.resize(img, (300, 300))                    # H,W
img_t    = TF.to_tensor(img).float()                     # (3,H,W) 0–1

prediction = detector.predict(img_t)                     # dict with boxes …
# print(prediction)

# 5) visualise -------------------------------------------------------
visulize(img, prediction, labels)                        # same helper