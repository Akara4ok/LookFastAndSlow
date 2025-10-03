from torchvision import transforms
import cv2
import numpy as np
import random
import torch

def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    max = np.minimum(a[:, 2:], b[2:])
    min = np.maximum(a[:, :2], b[:2])
    inter = np.clip((max - min), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    intersection = intersect(a, b)
    area_a = ((a[:, 2]-a[:, 0]) *
              (a[:, 3]-a[:, 1]))
    area_b = ((b[2]-b[0]) *
              (b[3]-b[1]))
    union = area_a + area_b - intersection
    return intersection / union

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tgt):
        for t in self.transforms:
            img, tgt = t(img, tgt)
        return img, tgt

class RandomSaturation():
    def __init__(self, lower_bound: float = 0.5, upper_bound: float = 1.5):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, img: np.ndarray, tgt: dict):
        if np.random.randint(2):
            img[:, :, 1] *= np.random.uniform(self.lower_bound, self.upper_bound)

        return img, tgt


class RandomHue():
    def __init__(self, value: float = 18.0):
        self.value = value

    def __call__(self, img: np.ndarray, tgt: dict):
        if np.random.randint(2):
            img[:, :, 0] += np.random.uniform(-self.value, self.value)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img, tgt

class SwapChannels():
    def __init__(self, swaps: list):
        self.swaps = swaps

    def __call__(self, img: np.ndarray, tgt: dict):
        img = img[:, :, self.swaps]
        return img, tgt

class RandomLightingNoise():
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img: np.ndarray, tgt: dict):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img, tgt = shuffle(img, tgt)
        return img, tgt


class ConvertColor():
    def __init__(self, cur: str, new: str):
        self.new = new
        self.cur = cur

    def __call__(self, img: np.ndarray, tgt: dict):
        if self.cur == 'BGR' and self.new == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.cur == 'RGB' and self.new == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.cur == 'BGR' and self.new == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.cur == 'HSV' and self.new == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        elif self.cur == 'HSV' and self.new == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img, tgt


class RandomContrast():
    def __init__(self, lower_bound: float = 0.5, upper_bound: float = 1.5):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, img: np.ndarray, tgt: dict):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower_bound, self.upper_bound)
            img *= alpha
        return img, tgt


class RandomBrightness():
    def __init__(self, delta: float = 32):
        self.delta = delta

    def __call__(self, img: np.ndarray, tgt: dict):
        if np.random.randint(2):
            value = np.random.uniform(-self.delta, self.delta)
            img += value
        return img, tgt

class RandomMirror():
    def __call__(self, img: np.ndarray, tgt: dict):
        _, width, _ = img.shape
        if np.random.randint(2):
            img = img[:, ::-1].copy()
            bboxes = tgt["boxes"].copy()
            bboxes[:, 0::2] = width - bboxes[:, 2::-2]
            tgt["boxes"] = bboxes
        return img, tgt

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, img: np.ndarray, tgt: dict):
        if np.random.randint(2):
            return img, tgt

        height, width, depth = img.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, width*ratio - width)
        top = np.random.uniform(0, height*ratio - height)

        expand_image = np.zeros((int(height*ratio), int(width*ratio), depth), dtype=img.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = img
        img = expand_image

        bboxes = tgt["boxes"].copy()
        bboxes[:, :] += (int(left), int(top), int(left), int(top))
        tgt["boxes"] = bboxes

        return img, tgt
    
class ToNormalizedCoords():
    def __call__(self, img: np.ndarray, tgt: dict):
        height, width, _ = img.shape
        tgt["boxes"][:, 0] /= width
        tgt["boxes"][:, 2] /= width
        tgt["boxes"][:, 1] /= height
        tgt["boxes"][:, 3] /= height

        return img, tgt
    
class PhotometricDistort():
    def __init__(self):
        self.transform = [
            RandomContrast(),
            ConvertColor("RGB", 'HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor('HSV', 'RGB'),
            RandomContrast()
        ]
        self.bright = RandomBrightness()
        self.light = RandomLightingNoise()

    def __call__(self, img: np.ndarray, tgt: dict):
        img = img.copy()
        img, tgt = self.bright(img, tgt)
        if np.random.randint(2):
            distort = Compose(self.transform[:-1])
        else:
            distort = Compose(self.transform[1:])
        img, tgt = distort(img, tgt)
        return self.light(img, tgt)

class RandomSampleCrop():
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def __call__(self, img: np.ndarray, tgt: np.ndarray):
        height, width, _ = img.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, tgt

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                current_image = img

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                overlap = jaccard(tgt["boxes"], rect)

                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                centers = (tgt["boxes"][:, :2] + tgt["boxes"][:, 2:]) / 2.0

                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                if not mask.any():
                    continue

                current_boxes = tgt["boxes"][mask, :].copy()
                current_labels = tgt["labels"][mask]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]
                
                tgt["boxes"] = current_boxes
                tgt["labels"] = current_labels

                return current_image, tgt

class ResizeNormalize():
    def __init__(self, size: int):
        self.size = size
        
    def __call__(self, img: np.ndarray, tgt: dict):
        img, tgt = ToNormalizedCoords()(img, tgt)
        img = (img).astype(np.uint8)
        img = transforms.ToTensor()(img)
        img = transforms.Resize((self.size, self.size))(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])(img)

        tgt["boxes"] = torch.tensor(tgt["boxes"], dtype=torch.float32)
        tgt["labels"] = torch.tensor(tgt["labels"], dtype=torch.int64)

        return img, tgt
