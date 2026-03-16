import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from Dataset.voc_dataset import VOCDataset
from Dataset.single_video_dataset import SingleVideoDataset
from Dataset.Yolo.YoloDataset import YoloTestDataset
import numpy as np
import matplotlib.pyplot as plt


voc_ds = VOCDataset("Data/VOCDevKitTest", "2007", "test", use_cache=False)
ds = YoloTestDataset(voc_ds, 340)

for img, tgt in ds:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    boxes = tgt["boxes"] 
    labels = tgt["labels"]
    img_width = img.shape[2]
    img_height = img.shape[1]

    for (box, label) in zip(boxes, labels): 
        xmin, ymin, xmax, ymax = box
        xmin *= img_width
        ymin *= img_height
        xmax *= img_width
        ymax *= img_height
        h = ymax - ymin
        w = xmax - xmin
        rect = plt.Rectangle((xmin, ymin), w , h,
                            fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{label}", color='red', fontsize=8)
    
    img_show = img.permute(1, 2, 0).cpu().numpy()
    img_show = np.clip(img_show, 0.0, 1.0)
    ax.imshow(img_show)

    plt.show()





