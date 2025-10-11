import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ConfigUtils.config import Config
from pathlib import Path
import logging
from Dataset.voc_dataset import VOCDataset
from Dataset.SSDLite.youtube_bb_dataset import YoutubeBBDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.SSDLite.map_labels_wrapper import MapLabelsWrapper, UNIFIED_CLASS_NAMES
from Dataset.SSDLite.mixed_seq_dataset import MixedSeqDataset
from ObjectDetector.SSDLite.phase1_trainer import Phase1Trainer
from ObjectDetector.SSDLite.video_object_detector import VideoObjectDetector
from ObjectDetector.SSDLite.Anchors.mobilenet_anchors import specs

logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['lr']['initial_lr'] = 0.001
config['model']['path'] = "/kaggle/working/Artifacts/Model/model.weights.h5"
config['model']['pretrain'] = "/kaggle/input/phase1pretrain/pytorch/default/1/phase1.weights.h5"
config['model']['fast_width'] = 0.5
config['train']['tensorboard_path'] = "/kaggle/working/Artifacts/Logs/"
config['train']['batch_size'] = 8
config['train']['epochs'] = 20
config['train']['augmentation'] = False
config['anchors']['iou_threshold'] = 0.45
config['anchors']['post_iou_threshold'] = 0.2
config['anchors']['confidence'] = 0.5
config['anchors']['top_k_classes'] = 200

voc_ds = VOCDataset("/kaggle/input/vocdataset/VOCdevkit", "2007", "trainval", use_cache=True)
voc_ds = ImageSeqVideoDataset(voc_ds)
voc_ds = MapLabelsWrapper(voc_ds, "voc")

yt_ds = YoutubeBBDataset(
    root="data/ytbb",
    split="train",            # or "val"
    frames_per_clip=6,        # returns 6 frames per sample
    download=True,            # allow downloading CSVs and videos
)
yt_ds = MapLabelsWrapper(yt_ds, "ytbb")

train_ds = MixedSeqDataset(300, voc_ds, yt_ds, 2500, (3, 1), 20, 100)
logging.info("dataset cached")

labels = list(UNIFIED_CLASS_NAMES.values())

objectDetector = VideoObjectDetector(labels, config, specs)
objectDetector.train(train_ds)
