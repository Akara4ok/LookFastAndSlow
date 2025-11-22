import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ConfigUtils.config import Config
from Dataset.voc_dataset import VOCDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqTestDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
config['train']['batch_size'] = 8
batch_size = config['train']['batch_size']

voc_ds = VOCDataset("Data/VOCdevkit", "2007", "trainval", use_cache=False)
assert len(voc_ds) > 0, "VOC dataset cannot be empty"

voc_seq = ImageSeqVideoDataset(voc_ds)
voc_seq = YoloSeqDataset(voc_seq, 640)

seq_len = len(voc_seq)
assert seq_len > 0, "Video sequence dataset must not be empty"

detector = CustomVideoObjectDetector(config, VOCDataset.VOC_CLASSES)
loader = detector._make_loader(voc_seq, shuffle=False)

batch = next(iter(loader))

imgs = batch["images"]
assert isinstance(imgs, object), "Batch must contain 'images'"
assert imgs.ndim == 5, "Images must have shape [B, T, C, H, W]"

B, T, C, H, W = imgs.shape
assert B == batch_size, f"Batch size must be {batch_size}"
assert T > 0, "Sequence length must be > 0"
assert C == 3, "Images must be RGB"
assert H == 640 and W == 640, "Images must be resized to 640x640"

boxes = batch["boxes"]
assert len(boxes) == B, "There must be one boxes-list per batch element"
assert len(boxes[0]) == T, "There must be one boxes-list per frame"

first_frame_boxes = boxes[0][0]
assert first_frame_boxes.ndim == 2, "Frame bbox must be a matrix [N,4]"
assert first_frame_boxes.shape[1] == 4, "Bounding box must have 4 coords"

labels = batch["labels"]
assert len(labels) == B, "One labels list per batch element"
assert len(labels[0]) == T, "One labels list per frame"

first_frame_labels = labels[0][0]
assert first_frame_labels.ndim == 1, "Frame labels must be 1D tensor"

ultra_batch = detector._prep_batch_for_ultra_video(batch)

uimgs = ultra_batch["imgs"]
assert uimgs.shape == (B, T, 3, 640, 640), "Ultra imgs must match original dimensions"

gt = ultra_batch["gt"]
assert len(gt) == T, "Ultra gt must have one dict per frame"

first_frame_gt = gt[0]

assert first_frame_gt["imgs"].shape == (B, 3, 640, 640), \
    "GT frame must contain stacked images as [B, C, H, W]"

bboxes = first_frame_gt["bboxes"]
assert bboxes.ndim == 2, "Ultra bboxes must be a matrix [N,4]"
assert bboxes.shape[1] == 4, "Each bbox must be [x1,y1,x2,y2]"

# labels
cls = first_frame_gt["cls"]
assert cls.ndim == 2, "GT labels must be shape [N,1]"
assert cls.shape[1] == 1, "Each label must have shape [1]"

# batch_idx
batch_idx = first_frame_gt["batch_idx"]
assert batch_idx.ndim == 1, "batch_idx must be 1D"
assert batch_idx.shape[0] == bboxes.shape[0] == cls.shape[0], \
    "batch_idx, bboxes, cls must have same num entries"

print("✔ All detection loader tests passed!")