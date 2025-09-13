from typing import Optional, Tuple, List
import random
import numpy as np
from torch.utils.data import Dataset, get_worker_info

class MixedSeqDataset(Dataset):
    def __init__(
        self,
        image_seq_ds: Dataset,
        video_ds: Dataset,
        num_samples: int,
        ratio: Tuple[int, int] = (1, 1),
        retries_per_item: int = 20,
        video_index_pool_size: Optional[int] = None,
        seed: int = 42,
    ):
        assert num_samples > 0
        assert ratio[0] >= 0 and ratio[1] >= 0 and (ratio[0] + ratio[1]) > 0

        self.image_seq_ds = image_seq_ds
        self.video_ds = video_ds
        self.num_samples = int(num_samples)
        self.img_weight, self.vid_weight = int(ratio[0]), int(ratio[1])
        self.retries_per_item = int(retries_per_item)
        self.base_seed = int(seed)

        self.p_img = self.img_weight / (self.img_weight + self.vid_weight)

        self._video_indices: List[int]
        n_vid = len(self.video_ds)
        if video_index_pool_size is None or video_index_pool_size >= n_vid:
            self._video_indices = list(range(n_vid))  # full range
        else:
            rng = random.Random(self.base_seed ^ 0xA5A5A5)
            self._video_indices = rng.sample(range(n_vid), k=video_index_pool_size)

        self._image_indices = list(range(len(self.image_seq_ds)))

    def __len__(self) -> int:
        return self.num_samples

    def _rng_for_idx(self, idx: int) -> random.Random:
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id + 1  # avoid colliding with 0
        seed = (self.base_seed * 1315423911) ^ (idx * 2654435761) ^ (wid * 97531)
        return random.Random(seed & 0xFFFFFFFF)

    def _sample_image_item(self, rng: random.Random):
        for _ in range(self.retries_per_item):
            i = rng.choice(self._image_indices)
            try:
                item = self.image_seq_ds[i]
                if item is None:
                    continue
                frames, targets = item
                total = sum(t["labels"].size for t in targets)
                if total == 0:
                    continue
                return frames, targets
            except IndexError:
                continue
        raise IndexError("Image-seq: failed to sample a valid item")

    def _sample_video_item(self, rng: random.Random):
        for _ in range(self.retries_per_item):
            i = rng.choice(self._video_indices)
            try:
                item = self.video_ds[i]
                if item is None:
                    continue
                frames, targets = item
                total = sum(t["labels"].size for t in targets)
                if total == 0:
                    continue
                return frames, targets
            except IndexError:
                continue
        raise IndexError("Video: failed to sample a valid item")

    def __getitem__(self, idx: int):
        rng = self._rng_for_idx(idx)

        choose_img = (rng.random() < self.p_img)

        first = "img" if choose_img else "vid"
        second = "vid" if choose_img else "img"

        for phase in (first, second):
            try:
                if phase == "img":
                    return self._sample_image_item(rng)
                else:
                    return self._sample_video_item(rng)
            except IndexError:
                continue

        raise IndexError("MixedSeqDataset: failed to draw a valid sample from either source")