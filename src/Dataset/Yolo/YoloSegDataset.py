import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from Dataset.augmentation import ToNormalizedCenterCoords, ToNormalizedCoords

class InferenceTransform:
    def __init__(self, size: int):
        self.size = size
        cv2.setUseOptimized(True)
        cv2.setNumThreads(8)

    def letterbox(self, img: np.ndarray):
        h0, w0 = img.shape[:2]
        s = self.size

        scale = min(s / h0, s / w0)
        nh, nw = int(h0 * scale), int(w0 * scale)

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_w = s - nw
        pad_h = s - nh
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))

        letterbox_params = {
            "scale": scale,
            "pad": (left, top),
            "new_shape": (nw, nh),
        }

        return padded, letterbox_params

    def __call__(self, img: np.ndarray):
        img, self.lb_params = self.letterbox(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        img = img.float().div_(255.0)
        return img.to("cuda", non_blocking=True), self.lb_params

            

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
    