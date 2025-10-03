import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Dataset.voc_dataset import VOCDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset

import numpy as np

voc_ds = VOCDataset("Data/VOCDevKit", "2007", "trainval", 300, use_cache=False)
ds = ImageSeqVideoDataset(voc_ds)

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

