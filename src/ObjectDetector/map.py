# ObjectDetector/metrics/map.py
from __future__ import annotations
from typing import Dict, List

import torch
import numpy as np

def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of corner-format boxes (xmin, ymin, xmax, ymax)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)

def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()

class MeanAveragePrecision:
    def __init__(self,
                 num_classes: int,
                 iou_threshold: float = 0.5,
                 device: str | torch.device = "cpu"):
        self.C = num_classes
        self.iou = iou_threshold
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.detections: List[Dict] = []
        self.groundtruth: List[Dict] = []

    def update(self,
               preds: List[Dict[str, torch.Tensor]],
               targets: List[Dict[str, torch.Tensor]]):
        for p, t in zip(preds, targets):
            self.detections.append({
                "boxes":   p["boxes"].detach().cpu().numpy(),
                "scores":  p["scores"].detach().cpu().numpy(),
                "labels":  p["classes"].detach().cpu().numpy()
            })
            self.groundtruth.append({
                "boxes":  t["boxes"].detach().cpu().numpy(),
                "labels": t["labels"].detach().cpu().numpy()
            })

    def compute(self) -> Dict[str, float]:
        aps = []
        
        for cls in range(1, self.C):
            det_cls = []
            gt_per_img = {}

            total_gt = 0
            for img_id, (d, g) in enumerate(zip(self.detections,
                                                self.groundtruth)):
                m = d["labels"] == cls
                for box, score in zip(d["boxes"][m], d["scores"][m]):
                    det_cls.append((img_id, score, box))

                mask = g["labels"] == cls
                gt_per_img[img_id] = {
                    "boxes": g["boxes"][mask],
                    "det":   np.zeros(mask.sum(), dtype=bool)
                }
                total_gt += mask.sum()

            if total_gt == 0:
                continue

            if len(det_cls) == 0:
                aps.append(0.0)
                continue

            
            det_cls.sort(key=lambda x: x[1], reverse=True)
            
            tp = np.zeros(len(det_cls))
            fp = np.zeros(len(det_cls))

            for i, (img_id, score, box_p) in enumerate(det_cls):
                g = gt_per_img[img_id]
                if g is None or g["boxes"].size == 0:
                    fp[i] = 1
                    continue
                ious = _box_iou(box_p[None, :], g["boxes"])[0]
                best = np.argmax(ious)
                if ious[best] >= self.iou and not g["det"][best]:
                    tp[i] = 1
                    g["det"][best] = True
                else:
                    fp[i] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            
            recalls = tp_cum / total_gt
            precis = tp_cum / (tp_cum + fp_cum + 1e-6)
            ap = _ap_from_pr(recalls, precis)
            aps.append(ap)
            # TP = int(tp.sum()) if len(aps) else 0
            # FP = int(fp.sum()) if len(aps) else 0
            # print("================")
            # print("Class", cls)
            # print("GT", total_gt)
            # print("TP", TP)
            # print("FP", int(fp.sum()) if len(aps) else 0)
            # print("FN", max(total_gt - TP, 0))
            # print("Recall", TP / total_gt if total_gt > 0 else 0.0)
            # print("Precision ", TP / (TP + FP) if (TP + FP) > 0 else 0.0)
            # print("ap", ap)
            
        mAP = float(np.mean(aps)) if aps else 0.0
        out = {"mAP": mAP}
        for c, ap in enumerate(aps, 1):
            out[f"AP_{c}"] = ap
            # qwe
        return out