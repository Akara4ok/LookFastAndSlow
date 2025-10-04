import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Dataset.SSDLite.image_seq_dataset import ImageSeqDataset

dataset = ImageSeqDataset("Data/Cifar100", "cifar100", "train", 3, 300, True, False)

for seq, label in dataset:
    n = len(seq)
    fig, axs = plt.subplots(1, n, figsize=(12, 4))

    for i, img in enumerate(seq):
        axs[i].imshow(img.permute(1, 2, 0))
        axs[i].axis("off")
        axs[i].set_title("Image " + str(i))

    plt.show()

