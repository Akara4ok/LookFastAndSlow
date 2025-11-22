import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqDataset
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector


logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/Yolo/test11x.pt"
config['train']['batch_size'] = 1
config["data"]["test_percent"] = 0.01


objectDetector = CustomVideoObjectDetector(config, VOCDataset.VOC_CLASSES)
objectDetector.load_weights(None, "Model/Yolo/yolo11n_voc_2.pt", "Model/Yolo/yolo11x_voc_2.pt", True)

voc_ds = VOCDataset("Data/VOCdevkit", "2007", "trainval", use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)
voc_ds = YoloSeqDataset(voc_ds, 640)

freeze = {
    # "backbone": (1, None, None),
    # "temporal": (1, 10, None),
    # "head": (1, 1, None)
}

# objectDetector.train(voc_ds)
objectDetector.train(voc_ds, freeze_dict=freeze)
