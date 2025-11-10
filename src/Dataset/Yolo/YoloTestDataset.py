import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from Dataset.augmentation import ToNormalizedCoords
    
class ResizeNormalizeYoloTest():
    def __init__(self, size: int):
        self.size = size
        
    def __call__(self, img: np.ndarray, tgt: dict):
        img, tgt = ToNormalizedCoords()(img, tgt)
        img = (img).astype(np.uint8)
        img = transforms.ToTensor()(img)
        img = transforms.Resize((self.size, self.size))(img)
        
        tgt["boxes"] = torch.tensor(tgt["boxes"], dtype=torch.float32)
        tgt["labels"] = torch.tensor(tgt["labels"], dtype=torch.int64)

        return img, tgt


class YoloTestDataset(Dataset):
    def __init__(self, dataset: Dataset, img_size: int):
        super().__init__()
        self.dataset = dataset
        self.transforms = ResizeNormalizeYoloTest(img_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, tgt = self.dataset.__getitem__(idx)
        return self.transforms(img, tgt)
    