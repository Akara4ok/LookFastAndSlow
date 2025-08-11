from torch.utils.data import Dataset
from torchvision import transforms
from Dataset.augmentation import Compose, ResizeNormalize, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToNormalizedCoords

class TrainDataset(Dataset):
    def __init__(self, dataset: Dataset, img_size: int):
        super().__init__()
        self.dataset = dataset
        self.transforms = Compose([
            PhotometricDistort(),
            Expand([0.485, 0.456, 0.406]),
            RandomSampleCrop(),
            RandomMirror(),
            ResizeNormalize(img_size)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, tgt = self.dataset.__getitem__(idx)
        return self.transforms(img, tgt)
    