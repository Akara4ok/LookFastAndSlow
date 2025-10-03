from typing import List, Tuple, Optional, Dict

import math
import random
import numpy as np
import cv2

class BoxOps:
    @staticmethod
    def clip_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
        """Clip xyxy boxes to [0..w-1], [0..h-1]."""
        if boxes.size == 0:
            return boxes
        boxes = boxes.copy()
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    @staticmethod
    def area_xyxy(boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.float32)
        w = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        h = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        return w * h

    @staticmethod
    def intersect_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0:
            return a.copy()
        inter = np.zeros_like(a)
        inter[:, 0] = np.maximum(a[:, 0], b[0])
        inter[:, 1] = np.maximum(a[:, 1], b[1])
        inter[:, 2] = np.minimum(a[:, 2], b[2])
        inter[:, 3] = np.minimum(a[:, 3], b[3])
        return inter

    @staticmethod
    def crop_and_resize(img: np.ndarray, crop_xywh: Tuple[float, float, float, float],
                        out_size: Tuple[int, int]) -> np.ndarray:
        x, y, w, h = crop_xywh
        x0 = int(round(x))
        y0 = int(round(y))
        x1 = int(round(x + w))
        y1 = int(round(y + h))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
        crop = img[y0:y1, x0:x1]
        outW, outH = out_size
        if crop.size == 0:
            return np.zeros((outH, outW, 3), dtype=img.dtype)
        return cv2.resize(crop, (outW, outH), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def remap_boxes_after_crop_resize(
        boxes_xyxy: np.ndarray,
        crop_xywh: Tuple[float, float, float, float],
        out_size: Tuple[int, int]
    ) -> np.ndarray:
        if boxes_xyxy.size == 0:
            return boxes_xyxy.copy()

        x, y, w, h = crop_xywh
        shifted = boxes_xyxy.copy()
        shifted[:, [0, 2]] -= x
        shifted[:, [1, 3]] -= y

        shifted[:, 0] = np.clip(shifted[:, 0], 0, w)
        shifted[:, 1] = np.clip(shifted[:, 1], 0, h)
        shifted[:, 2] = np.clip(shifted[:, 2], 0, w)
        shifted[:, 3] = np.clip(shifted[:, 3], 0, h)

        outW, outH = out_size
        sx = outW / max(1e-6, w)
        sy = outH / max(1e-6, h)
        shifted[:, [0, 2]] *= sx
        shifted[:, [1, 3]] *= sy
        return shifted

class MotionPath:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def _pick_base_crop(self, img_w: int, img_h: int,
                        boxes: np.ndarray) -> Tuple[float, float, float, float]:
        if boxes.size > 0:
            # pick a ground-truth box as an anchor
            i = self.rng.randrange(len(boxes))
            x1, y1, x2, y2 = boxes[i]
            bw = max(4.0, (x2 - x1))
            bh = max(4.0, (y2 - y1))
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

            # pad the box to get a crop slightly larger than the object
            padw = bw * 0.3
            padh = bh * 0.3
            w = min(img_w, bw + 2 * padw)
            h = min(img_h, bh + 2 * padh)
            x = max(0.0, min(img_w - w, cx - 0.5 * w))
            y = max(0.0, min(img_h - h, cy - 0.5 * h))
            return (x, y, w, h)
        else:
            # global random crop covering 70–100% of the min dimension
            base = min(img_w, img_h)
            side = self.rng.uniform(0.7, 1.0) * base
            w = min(side * self.rng.uniform(0.9, 1.1), img_w)
            h = min(side * self.rng.uniform(0.9, 1.1), img_h)
            x = self.rng.uniform(0, max(1.0, img_w - w))
            y = self.rng.uniform(0, max(1.0, img_h - h))
            return (x, y, w, h)

    def build(self, img_w: int, img_h: int, boxes: np.ndarray, seq_len: int, max_translate: float) -> List[Tuple[float, float, float, float]]:
        crop = self._pick_base_crop(img_w, img_h, boxes)
        x, y, w, h = crop
        crops = [(x, y, w, h)]
        # cumulative scale vs base
        cum_scale = 1.0

        for _ in range(seq_len - 1):
            # translation proportional to current crop size
            tx = self.rng.uniform(-max_translate, max_translate) * w
            ty = self.rng.uniform(-max_translate, max_translate) * h

            # smooth zoom via log-normal-ish noise
            z = math.exp(self.rng.gauss(0.0, 0.06))
            cum_scale = max(0.6, min(1.3, cum_scale * z))
            # scale around crop center
            cx, cy = x + 0.5 * w, y + 0.5 * h
            new_w = w * (cum_scale)
            new_h = h * (cum_scale)

            # apply translation
            nx = cx - 0.5 * new_w + tx
            ny = cy - 0.5 * new_h + ty

            # keep inside image
            nx = max(0.0, min(img_w - new_w, nx))
            ny = max(0.0, min(img_h - new_h, ny))

            x, y, w, h = nx, ny, new_w, new_h
            crops.append((x, y, w, h))

        return crops


class SequenceSynthesizer:
    """
    Applies a motion path to (image, boxes, labels) → builds a T-frame clip.
    """

    def __init__(self, seq_len: int, max_translate: float, out_size: int, rng: Optional[random.Random] = None):
        self.out_size = (out_size, out_size)
        self.rng = rng or random.Random()
        self.seq_len = seq_len
        self.max_translate = max_translate

    def _visible_filter(self, boxes_before: np.ndarray, boxes_after: np.ndarray) -> np.ndarray:
        if boxes_before.size == 0:
            return np.zeros((0,), dtype=bool)
        a0 = BoxOps.area_xyxy(boxes_before)
        a1 = BoxOps.area_xyxy(boxes_after)
        vis = np.divide(a1, np.maximum(1e-6, a0), dtype=np.float32)
        return vis >= 0.4

    def synthesize_once(self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:
        H, W = img.shape[:2]
        path_builder = MotionPath(self.rng)
        crops = path_builder.build(W, H, boxes, self.seq_len, self.max_translate)

        frames: List[np.ndarray] = []
        targets: List[Dict[str, np.ndarray]] = []
        outW, outH = self.out_size

        for (x, y, w, h) in crops:
            # Remap boxes to crop coords and then to output size
            boxes_after = BoxOps.remap_boxes_after_crop_resize(boxes, (x, y, w, h), (outW, outH))
            # Determine visibility by intersecting original boxes with crop (pre-scale)
            inter_in_crop = BoxOps.intersect_xyxy(boxes, np.array([x, y, x + w, y + h], dtype=np.float32))
            keep = self._visible_filter(boxes, inter_in_crop)

            boxes_after = boxes_after[keep]
            labels_after = labels[keep] if labels.size > 0 else labels

            # If we must keep at least 1 box per frame and it's empty, we can optionally
            # recentre crop to anchor object (but better to rely on MotionConfig(anchor=random_box))
            if boxes_after.shape[0] == 0:
                # Leave empty for now; caller decides if whole sequence is acceptable.
                pass

            frame = BoxOps.crop_and_resize(img, (x, y, w, h), (outW, outH))
            frames.append(frame)
            targets.append({"boxes": boxes_after.astype(np.float32),
                            "labels": labels_after})

        return frames, targets

    def synthesize(self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray
                   ) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:
        for attempt in range(4):
            frames, targets = self.synthesize_once(img, boxes, labels)
            if all(t["boxes"].shape[0] > 0 for t in targets):
                return frames, targets
        return frames, targets
