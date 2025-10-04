from torch.utils.data import Dataset
from torchvision import transforms
from Dataset.SSDLite.augmentation import Compose, ResizeNormalize

class TestDataset(Dataset):
    def __init__(self, dataset: Dataset, img_size: int):
        super().__init__()
        self.dataset = dataset
        self.transforms = Compose([
            ResizeNormalize(size=img_size),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, tgt = self.dataset.__getitem__(idx)
        return self.transforms(img, tgt)
    