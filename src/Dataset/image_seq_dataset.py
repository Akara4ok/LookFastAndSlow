import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import List

DATASET_DEFAULTS = {
    "cifar10":  {"num_classes": 10,
                 "mean": (0.4914, 0.4822, 0.4465),
                 "std":  (0.2470, 0.2435, 0.2616)},
    "cifar100": {"num_classes": 100,
                 "mean": (0.5071, 0.4865, 0.4409),
                 "std":  (0.2673, 0.2564, 0.2762)},
    "stl10":    {"num_classes": 10,
                 "mean": (0.4467, 0.4398, 0.4066),
                 "std":  (0.2603, 0.2566, 0.2713)},
    "caltech101":{"num_classes": 101,
                 "mean": (0.485, 0.456, 0.406),
                 "std":  (0.229, 0.224, 0.225)},
    "food101":  {"num_classes": 101,
                 "mean": (0.545, 0.443, 0.349),
                 "std":  (0.252, 0.245, 0.260)},
}

def _build_transform(split: str, img_size: int, mean: list, std: list):
    aug: List = []
    if split == "train":
        aug += [
            transforms.Resize(int(img_size * 1.15)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        try:
            from torchvision.transforms import RandAugment
            aug += [RandAugment()]
        except Exception:
            pass
    else:
        aug += [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
        ]
    aug += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(aug)

class ImageSeqDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset: str = "cifar100",
        split: str = "train",
        seq_len: int = 3,
        img_size: int = 300,
        download: bool = True,
        use_cache = True
    ):
        super().__init__()
        ds_name = dataset.lower()
        assert ds_name in DATASET_DEFAULTS, f"Unsupported dataset: {ds_name}"
        self.seq_len = seq_len
        self.cfg = DATASET_DEFAULTS[ds_name]
        self.img_size = img_size

        self.transform = _build_transform(
            split="train" if split == "train" else "val",  # map "val"/"test" → eval augs
            img_size=img_size, mean=self.cfg["mean"], std=self.cfg["std"]
        )

        if ds_name == "cifar10":
            self.base = datasets.CIFAR10(root, train=(split=="train"), transform=None, download=download)
        elif ds_name == "cifar100":
            self.base = datasets.CIFAR100(root, train=(split=="train"), transform=None, download=download)
        elif ds_name == "stl10":
            assert split in ("train","test"), "STL10 split must be 'train' or 'test'."
            self.base = datasets.STL10(root, split=split, transform=None, download=download)
        elif ds_name == "caltech101":
            self.base = datasets.Caltech101(root, transform=None, download=download)
        elif ds_name == "food101":
            self.base = datasets.Food101(root, split="train" if split=="train" else "test",
                                         transform=None, download=download)
        else:
            raise ValueError("Unexpected dataset")

        self.num_classes = self.cfg["num_classes"]
        
        self.use_cache = use_cache
        self.is_cached = False
        self.cached_data = []
        
        if(self.use_cache):
            self._build_cache()
        
    def set_val_transform_type(self):
        self.transform = _build_transform(split="val", img_size=self.img_size, mean=self.cfg["mean"], std=self.cfg["std"])

    def __len__(self):
        return len(self.base)

    def _load_raw(self, idx):
        if(self.use_cache and self.is_cached):
            img, label = self.cached_data[idx]
            return img.copy(), label.copy()
        
        img, label = self.base[idx]
        return img, int(label)
    
    def _build_cache(self):
        for i in range(self.__len__()):
            self.__getitem__(i)
        self.is_cached = True

    def __getitem__(self, idx: int):            
        img, label = self._load_raw(idx)
        if(self.use_cache and not self.is_cached):
            self.cached_data((img.copy(), label.copy()))

        img_t = self.transform(img)
        imgs_seq = torch.stack([img_t]*self.seq_len, dim=0)

        return imgs_seq, label