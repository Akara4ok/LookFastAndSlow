import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ConfigUtils.config import Config
from pathlib import Path
import logging
from Dataset.SSDLite.xml_star_dataset import XMLStarDataset
from Dataset.SSDLite.train_dataset import TrainDataset
from Dataset.SSDLite.test_dataset import TestDataset
from Dataset.voc_dataset import VOCDataset
from ObjectDetector.SSDLite.image_object_detector import ImageObjectDetector
from ObjectDetector.SSDLite.Anchors.mobilenet_anchors import specs
from ssd_training.visualize import visulize          # unchanged helper

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/model.weights.h5"
config['data']['path'] = "Data/VOCDevKitTest"
config['train']['batch_size'] = 32
config['anchors']['post_iou_threshold'] = 0.2
config['anchors']['confidence'] = 0.6
config['anchors']['top_k_classes'] = 10

test_ds = VOCDataset("Data/VOCdevkit", "2007", "train", False)

img_to_test = 96

labels = [ "background" ] + VOCDataset.VOC_CLASSES
objectDetector = ImageObjectDetector(labels, config, specs)
objectDetector.load_weights(config["model"]["path"])

map = objectDetector.test(test_ds, img_to_test)
print(map)

