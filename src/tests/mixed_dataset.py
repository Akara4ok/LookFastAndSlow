import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Dataset.voc_dataset import VOCDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.youtube_bb_dataset import YoutubeBBDataset
from Dataset.map_labels_wrapper import MapLabelsWrapper
from Dataset.mixed_seq_dataset import MixedSeqDataset

import numpy as np

voc_ds = VOCDataset("Data/VOCDevKit", "2007", "trainval", 300, use_cache=False)
voc_ds = ImageSeqVideoDataset(voc_ds)
voc_ds = MapLabelsWrapper(voc_ds, "voc")


yt_ds = YoutubeBBDataset(
    root="data/ytbb",
    split="train",            # or "val"
    frames_per_clip=6,        # returns 6 frames per sample
    download=True,            # allow downloading CSVs and videos
)
yt_ds = MapLabelsWrapper(yt_ds, "ytbb")

ds = MixedSeqDataset(voc_ds, yt_ds, 500, (3, 1), 20)
# ds = voc_ds

for seq, tgt in ds:
    if(len(seq) == 0):
        continue
    n = len(seq)
    fig, axs = plt.subplots(1, n, figsize=(12, 4))

    for i, img in enumerate(seq):
        img_rgb = img[..., ::-1]
        img_rgb = img_rgb.astype(np.int32)
        axs[i].imshow(img_rgb)
        axs[i].axis("off")
        axs[i].set_title("Image " + str(i))

    print(tgt)
    plt.show()


