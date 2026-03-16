import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Dataset.voc_dataset import VOCDataset
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from Dataset.single_video_dataset import SingleVideoDataset

import numpy as np

voc_ds = VOCDataset("Data/VOCdevkit", "2007", "trainval", use_cache=False)
ds = SingleVideoDataset("Data/SingleVideo", 6)
# ds = ImageSeqVideoDataset(voc_ds)

skip = 0
for imgs, tgt in ds:
    if(skip < 5):
        skip+=1
        continue
    n = len(imgs)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for i, ax in enumerate(axes.flat):
        boxes = tgt[i]["boxes"] 
        labels = tgt[i]["labels"]

        for (box, label) in zip(boxes, labels): 
            xmin, ymin, xmax, ymax = box
            h = ymax - ymin
            w = xmax - xmin
            rect = plt.Rectangle((xmin, ymin), w , h,
                                fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{label}", color='red', fontsize=8)
        
        ax.axis("off")
        ax.set_title("Image " + str(i))
        ax.imshow(imgs[i] / 255)

    plt.show()

