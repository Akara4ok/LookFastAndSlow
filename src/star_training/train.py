import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ConfigUtils.config import Config
from pathlib import Path
import logging
from Dataset.xml_star_dataset import XMLStarDataset
from ObjectDetector.object_detector import ObjectDetector
from ObjectDetector.Anchors.mobilenet_anchors import specs

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/star.weights.h5"
config['train']['epochs'] = 100
config['data']['path'] = "Data/"
config['anchors']['iou_threshold'] = 0.45
config['anchors']['confidence'] = 0.5
config['anchors']['top_k_classes'] = 200

dataset = XMLStarDataset(config['data']['path'], config['model']['img_size'])

objectDetector = ObjectDetector(['None', 'Star'], config, specs)
objectDetector.train(dataset)