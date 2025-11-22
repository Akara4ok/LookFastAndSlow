import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import numpy as np

from Dataset.voc_dataset import VOCDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqDataset, YoloSeqTestDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector

voc_ds = VOCDataset("Data/VOCdevkit", "2007", "trainval", use_cache=False)

assert len(voc_ds) > 0, "VOC dataset must not be empty"
assert hasattr(voc_ds, "__getitem__"), "VOC dataset must support indexing"

voc_ds = ImageSeqVideoDataset(voc_ds)

train_ds = YoloSeqDataset(copy.deepcopy(voc_ds), img_size=640)
test_ds  = YoloSeqTestDataset(copy.deepcopy(voc_ds), img_size=640)

seq_len = len(voc_ds)
assert seq_len > 0, "Sequence dataset must contain sequences"

seq_idx = 0
imgs, tgts = voc_ds[seq_idx]

assert isinstance(imgs, list), "Sequence images must be a list"
assert isinstance(tgts, list), "Sequence targets must be a list"
assert len(imgs) == len(tgts), "Every frame must have its own target list"

assert len(imgs) > 0, "Sequence must contain at least one frame"

for frame_idx, img in enumerate(imgs):
    assert isinstance(img, np.ndarray), "Raw frames must be numpy arrays"
    assert img.ndim in (2, 3), f"Unexpected image ndim={img.ndim}"
    assert len(tgts[frame_idx]) >= 0, "Targets list must exist (may be empty)"


imgs_preprocessed, train_tgts_raw = train_ds[seq_idx]

assert isinstance(imgs_preprocessed, np.ndarray) or hasattr(imgs_preprocessed, "shape"), \
    "Preprocessed images must be tensor/ndarray"

assert len(imgs_preprocessed.shape) == 4, "Train dataset must return 4D tensor"
T, C, H, W = imgs_preprocessed.shape
assert C == 3, "Train images must be RGB"

assert len(train_tgts_raw) == T, "Train dataset must return a target list per frame"

for tgt in train_tgts_raw:
    assert isinstance(tgt, dict), "Targets in train_ds must be dictionaries"
    assert "boxes" in tgt, "Train target must contain 'boxes'"
    assert "labels" in tgt, "Train target must contain 'labels'"

frames_test, test_tgts = test_ds[seq_idx]

assert isinstance(frames_test, np.ndarray) or hasattr(frames_test, "shape"), \
    "Test images must be tensor/ndarray"

assert len(frames_test.shape) == 4, "Test dataset must return (T, C, H, W)"
Tt, C2, H2, W2 = frames_test.shape
assert C2 == 3, "Test images must be RGB"
assert len(test_tgts) == Tt, "Test targets count must match number of frames"

for tgt in test_tgts:
    assert isinstance(tgt, dict), "Test target must be a dict"
    assert "boxes" in tgt, "Test target must contain boxes"
    assert "labels" in tgt, "Test target must contain labels"

assert len(imgs) == imgs_preprocessed.shape[0], "Train seq length must match raw length"
assert len(imgs) == frames_test.shape[0], "Test seq length must match raw length"

assert imgs_preprocessed.dtype in (np.float32, np.float64) or "float" in str(imgs_preprocessed.dtype), \
    "Train images must be float tensors"

assert imgs_preprocessed.max() <= 5, "Preprocessed images should not explode in magnitude"
assert imgs_preprocessed.min() >= -5, "Preprocessed images should not be too negative"

for t in train_tgts_raw:
    if t["boxes"].shape[0] > 0:
        assert t["boxes"].shape[1] == 4, "Each bbox must have shape (4,)"

for t in test_tgts:
    if t["boxes"].shape[0] > 0:
        assert t["boxes"].shape[1] == 4

print("✔ All dataset tests passed!")