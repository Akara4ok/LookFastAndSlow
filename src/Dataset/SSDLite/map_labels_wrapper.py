from typing import Tuple
import numpy as np
from torch.utils.data import Dataset

YT_ALIAS = {
    0: 12,
    1: 3,
    2: 2,
    3: 4,
    4: 5,
    6: 8,
    7: 7,
    9: 13,
    10: 10,
    11: 11,
    13: 1,
    15: 14,
    19: 9,
    22: 6
}

VOC_ALIAS = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    11: 9,
    12: 10,
    13: 11,
    14: 12,
    15: 13,
    18: 14
}

UNIFIED_CLASS_NAMES = {
    0: 'background',
    1: 'aeroplane', 
    2: 'bicycle', 
    3: 'bird', 
    4: 'boat', 
    5: 'bus', 
    6: 'car', 
    7: 'cat', 
    8: 'cow', 
    9: 'dog', 
    10: 'horse', 
    11: 'motorbike', 
    12: 'person', 
    13: 'pottedplant', 
    14: 'train'
}

class MapLabelsWrapper(Dataset):
    def __init__(self, base: Dataset, source: str):
        self.base = base
        self.source = source

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if(item is None):
            return None
        frames, targets = item
        mapped_targets = []
        return_empty = False
        for t in targets:
            boxes = np.asarray(t["boxes"], dtype=np.float32)
            labels = np.asarray(t["labels"], dtype=np.int64)

            mapped_labels, keep_mask = self.map_labels_array(labels, self.source)
            if keep_mask.size > 0:
                boxes = boxes[keep_mask]
            else:
                return_empty = True
                break

            mapped_targets.append({
                "boxes": boxes.astype(np.float32),
                "labels": mapped_labels.astype(np.int64),
            })
            
        if(return_empty):
            return None
        
        return frames, mapped_targets
    
    def map_labels_array(self, labels: np.ndarray, source: str) -> Tuple[np.ndarray, np.ndarray]:
        if source == "voc":
            mapping = VOC_ALIAS
        elif source == "ytbb":
            mapping = YT_ALIAS
        else:
            raise ValueError("source must be 'voc' or 'ytbb'")

        mapped = []
        keep = []
        for lbl in labels.tolist():
            if lbl in mapping:
                mapped.append(mapping[lbl])
                keep.append(True)
            else:
                keep.append(False)
                
        if len(mapped) == 0:
            return np.zeros((0,), dtype=np.int64), np.array(keep, dtype=bool)
        return np.asarray(mapped, dtype=np.int64), np.asarray(keep, dtype=bool)