from ConfigUtils.config import Config
from pathlib import Path
import logging
from Dataset.xml_star_dataset import XMLStarDataset
from ObjectDetector.object_detector import ObjectDetector
from ObjectDetector.Anchors.mobilenet_anchors import specs
from torchsummary import summary

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()

dataset = XMLStarDataset(config['data']['path'], config['model']['img_size'])

objectDetector = ObjectDetector(['None', 'Star'], config, specs)

summary(objectDetector.model, (3, 300, 300))


