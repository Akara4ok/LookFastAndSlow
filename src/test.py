from torch.utils.data import DataLoader
from Dataset.voc_dataset import VOCDataset

# first call downloads & extracts automatically
voc_train = VOCDataset("Data/VOCDevKit", "2007", "trainval", 300)

loader = DataLoader(voc_train, batch_size=8,
                    shuffle=True,
                    collate_fn=lambda b: list(zip(*b)))

for images, targets in loader:
    print(images[0].shape, targets[0]["boxes"].shape)
