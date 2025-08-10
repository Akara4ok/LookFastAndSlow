from ConfigUtils.config import Config
from pathlib import Path
import logging
from Dataset.xml_star_dataset import XMLStarDataset
from Dataset.voc_dataset import VOCDataset
from ObjectDetector.object_detector import ObjectDetector
from ObjectDetector.Anchors.mobilenet_anchors import specs

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/voc.weights.h5"
config['train']['epochs'] = 100
config['data']['path'] = "Data/VOCDevKit"
config['anchors']['iou_threshold'] = 0.45
config['anchors']['confidence'] = 0.5
config['anchors']['top_k_classes'] = 200

train_ds = VOCDataset("Data/VOCDevKit", "2007", "trainval", 300)

objectDetector = ObjectDetector([ "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
], config, specs)
objectDetector.train(train_ds)
