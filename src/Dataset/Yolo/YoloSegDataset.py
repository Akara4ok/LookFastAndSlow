import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from Dataset.augmentation import ToNormalizedCenterCoords, ToNormalizedCoords

class ResizeNormalizeSeqYolo():
    def __init__(self, size: int, normalize: ToNormalizedCoords):
        self.size = size
        self.normalize = normalize
        
    def __call__(self, imgs: list[np.ndarray], tgts: list[dict]):
        imgs_out = []
        tgts_out = []


        for img, tgt in zip(imgs, tgts):
            img, tgt = self.normalize(img, tgt)
            img = (img).astype(np.uint8)
            img = transforms.ToTensor()(img)
            img = transforms.Resize((self.size, self.size))(img)
            imgs_out.append(img)

            tgts_out.append({
                "boxes": torch.tensor(tgt["boxes"], dtype=torch.float32),
                "labels": torch.tensor(tgt["labels"], dtype=torch.int64),
            })

        imgs_out = torch.stack(imgs_out)

        return imgs_out, tgts_out

class YoloSeqDataset(Dataset):
    def __init__(self, dataset: Dataset, img_size: int):
        super().__init__()
        self.dataset = dataset
        self.transforms = ResizeNormalizeSeqYolo(img_size, ToNormalizedCenterCoords())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, tgt = self.dataset.__getitem__(idx)
        return self.transforms(img, tgt)
    
class YoloSeqTestDataset(Dataset):
    def __init__(self, dataset: Dataset, img_size: int):
        super().__init__()
        self.dataset = dataset
        self.transforms = ResizeNormalizeSeqYolo(img_size, ToNormalizedCoords())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, tgt = self.dataset.__getitem__(idx)
        return self.transforms(img, tgt)
    