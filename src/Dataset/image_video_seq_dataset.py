import random
import numpy as np
from torch.utils.data import Dataset
from Dataset.image_video_seq_utilites import SequenceSynthesizer

class ImageSeqVideoDataset(Dataset):
    def __init__(self,
                 base_dataset: Dataset,
                 seq_len: int = 6,
                 out_size: int = 300,
                 max_translate: float = 0.12,
                 rng_seed: int = 42):
        self.base = base_dataset
        self.rng = random.Random(rng_seed)
        self.seq_len = seq_len
        self.max_translate = max_translate
        self.out_size = out_size
        self.synth = SequenceSynthesizer(self.seq_len, self.max_translate, self.out_size, self.rng)

    def __len__(self) -> int:
        return len(self.base)

    def _unpack_base_sample(self, sample):
        if isinstance(sample, tuple) and len(sample) >= 2:
            img, target = sample[0], sample[1]
        elif isinstance(sample, dict):
            img, target = sample["image"], sample["target"]
        else:
            raise TypeError("Unsupported base_dataset sample format")

        boxes = target.get("boxes", None)
        labels = target.get("labels", None)
        if boxes is None or labels is None:
            raise KeyError("Base target must contain 'boxes' and 'labels'")

        # Ensure numpy arrays
        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels)
        return img, boxes, labels

    def __getitem__(self, idx: int):
        base_sample = self.base[idx]
        img, boxes, labels = self._unpack_base_sample(base_sample)

        frames, targets = self.synth.synthesize(img, boxes, labels)

        return frames, targets
