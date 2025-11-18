import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from Dataset.augmentation import Letterbox, LetterboxRemapBox

class ResizeNormalizeYolo():
    def __init__(self, size: int, return_xyxy: bool):
        self.size = size
        self.letterbox = Letterbox(size=640)
        self.remap = LetterboxRemapBox(return_xyxy)
        
    def __call__(self, img: np.ndarray, tgt: dict):
        img_lb, r, pad = self.letterbox(img)
        boxes_norm = self.remap(tgt["boxes"], r, pad, out_size=640)

        img = (img_lb).astype(np.uint8)
        img = transforms.ToTensor()(img)
        
        tgt["boxes"] = torch.tensor(boxes_norm, dtype=torch.float32)
        tgt["labels"] = torch.tensor(tgt["labels"], dtype=torch.int64)

        return img, tgt

class YoloDataset(Dataset):
    def __init__(self, dataset: Dataset, img_size: int):
        super().__init__()
        self.dataset = dataset
        self.transforms = ResizeNormalizeYolo(img_size, False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, tgt = self.dataset.__getitem__(idx)
        return self.transforms(img, tgt)

class YoloTestDataset(Dataset):
    def __init__(self, dataset: Dataset, img_size: int):
        super().__init__()
        self.dataset = dataset
        self.transforms = ResizeNormalizeYolo(img_size, True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, tgt = self.dataset.__getitem__(idx)
        return self.transforms(img, tgt)
    