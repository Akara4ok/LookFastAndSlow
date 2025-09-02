import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ConfigUtils.config import Config
from pathlib import Path
import logging
from Dataset.image_seq_dataset import ImageSeqDataset
from ObjectDetector.phase1_trainer import Phase1Trainer

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['model']['path'] = "Model/phase1.weights.h5"
config['train']['epochs'] = 100

train_ds = ImageSeqDataset("Data/Cifar10", "cifar10", "train", 3, 300, True, False)
val_ds = ImageSeqDataset("Data/Cifar10", "cifar10", "train", 3, 300, True, False)

logging.info("dataset cached")

phase1_trainer = Phase1Trainer(10, config)
phase1_trainer.train(train_ds, val_ds)
