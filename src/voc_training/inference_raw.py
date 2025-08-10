import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

from ConfigUtils.config import Config
from ObjectDetector.object_detector import ObjectDetector
from visualize import visulize          # unchanged helper
from ObjectDetector.Anchors.mobilenet_anchors import specs

logging.basicConfig(level=logging.INFO)

cfg = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
cfg['model']['path'] = "Model/voc.weights.h5"
cfg['anchors']['post_iou_threshold'] = 0.2
cfg['anchors']['confidence'] = 0.4
cfg['anchors']['top_k_classes'] = 200

labels = [ "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]                      # background idx 0 + 1 class

detector = ObjectDetector(labels, cfg, specs)

detector.load_weights(cfg["model"]["path"])

img_path = Path("Data/voc_test/17.jpg")
img      = Image.open(img_path).convert("RGB")           # PIL Image
img = img.resize((300, 300))

transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225]),
        ])
img_t = transform(img)

prediction = detector.predict(img_t)                     # dict with boxes …
print(prediction)

# 5) visualise -------------------------------------------------------
visulize(img, prediction, labels)                        # same helper