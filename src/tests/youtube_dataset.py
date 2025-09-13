import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Dataset.youtube_bb_dataset import YoutubeBBDataset

# Creates folder, downloads annotations if missing, then lazily downloads videos as needed.
ds = YoutubeBBDataset(
    root="data/ytbb",
    split="train",            # or "val"
    frames_per_clip=6,        # returns 6 frames per sample
    download=True,            # allow downloading CSVs and videos
)

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

