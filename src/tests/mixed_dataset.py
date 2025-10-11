import os
import sys
import matplotlib.pyplot as plt
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Dataset.voc_dataset import VOCDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.SSDLite.youtube_bb_dataset import YoutubeBBDataset
from Dataset.SSDLite.map_labels_wrapper import MapLabelsWrapper
from Dataset.SSDLite.mixed_seq_dataset import MixedSeqDataset

import numpy as np

# logging.basicConfig(level=logging.DEBUG)

voc_ds = VOCDataset("Data/VOCDevKit", "2007", "trainval", use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)
voc_ds = MapLabelsWrapper(voc_ds, "voc")


yt_ds = YoutubeBBDataset(
    root="data/ytbb",
    split="train",            # or "val"
    frames_per_clip=6,        # returns 6 frames per sample
    download=True,            # allow downloading CSVs and videos
)
yt_ds = MapLabelsWrapper(yt_ds, "ytbb")

ds = MixedSeqDataset(300, voc_ds, yt_ds, 500, (3, 1), 20, 10)
# ds = voc_ds

for seq, tgt in ds:
    if(len(seq) == 0):
        continue
    n = len(seq)
    fig, axs = plt.subplots(1, n, figsize=(12, 4))

    for i, img in enumerate(seq):
        img = img.permute(1, 2, 0)
        img = img.numpy() * 255
        img_rgb = img[..., ::-1]
        img_rgb = img_rgb.astype(np.int32)
        axs[i].imshow(img_rgb)
        axs[i].axis("off")
        axs[i].set_title("Image " + str(i))

    print(tgt)
    print("=" * 10)
    plt.show()


