import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy

from Dataset.voc_dataset import VOCDataset
from Dataset.Yolo.YoloSegDataset import YoloSeqDataset, YoloSeqTestDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector

import matplotlib.pyplot as plt

voc_ds = VOCDataset("Data/VOCdevkit", "2007", "trainval", use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)
train_ds = YoloSeqDataset(copy.deepcopy(voc_ds), 640)
test_ds = YoloSeqTestDataset(copy.deepcopy(voc_ds), 640)

print("Number of sequences in VOC 2007 trainval:", len(voc_ds))
for seq_idx in range(1):
    imgs, tgts = voc_ds[seq_idx]
    print(f"Sequence {seq_idx} length: {len(imgs)}")

    n = len(imgs)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i] / 255)
        ax.axis("off")
        ax.set_title("Image " + str(i))

    plt.show()

    for frame_idx, (img, tgt) in enumerate(zip(imgs, tgts)):
        print(f"  Frame {frame_idx} image shape: {img.shape}, targets: {tgt}")

print("="*125)


for seq_idx in range(1):
    frames, train_tgts = train_ds[seq_idx]
    print(f"Shape: {frames.shape}, Number of targets: {len(train_tgts)}")
    for frame_idx, (img, tgt) in enumerate(zip(frames, train_tgts)):
        print(f"  Frame {frame_idx} image shape: {img.shape}, targets: {tgt}")

print("="*125)

for seq_idx in range(1):
    frames, tgts = test_ds[seq_idx]
    print(f"Shape: {frames.shape}, Number of targets: {len(tgts)}")
    for frame_idx, (img, tgt) in enumerate(zip(frames, tgts)):
        print(f"  Frame {frame_idx} image shape: {img.shape}, targets: {tgt}")

print("="*125)
