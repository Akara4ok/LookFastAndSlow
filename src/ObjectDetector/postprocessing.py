from typing import Dict

import torch
import torch.nn.functional as F
from torchvision.ops import nms
from ObjectDetector.Anchors.anchors import Anchors

class PostProcessor:
    def __init__(self,
                 anchors: Anchors,
                 conf_thresh: float = 0.5,
                 iou_thresh:  float = 0.5,
                 top_k: int = 1):
        self.anchors = anchors
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.top_k = top_k

    def ssd_postprocess(self, cls_logits: torch.Tensor, pred_loc: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores = F.softmax(cls_logits, dim=-1)
        variances = self.anchors.variances
        boxes = Anchors.decode_boxes(pred_loc, self.anchors.center_anchors, variances[:2], variances[2:])
        boxes = Anchors.center_to_corner(boxes)
        boxes = boxes.clamp(0, 1)  # keep inside image
        all_boxes = []
        all_scores = []
        all_labels = []

        num_classes = scores.size(1)

        for c in range(1, num_classes):
            cls_scores = scores[:, c]
            keep = cls_scores > self.conf_thresh
            if not keep.any():
                continue

            cls_boxes  = boxes[keep]
            cls_scores = cls_scores[keep]
            keep_idx = nms(cls_boxes, cls_scores, self.iou_thresh)
            keep_idx = keep_idx[: self.top_k]            # top-k per class

            all_boxes.append(cls_boxes[keep_idx])
            all_scores.append(cls_scores[keep_idx])
            all_labels.append(torch.full((keep_idx.numel(),),
                                          c,
                                          dtype=torch.int32,
                                          device=cls_boxes.device))
        if not all_boxes:
            empty = torch.empty((0, 4), device=boxes.device)
            return dict(boxes=empty,
                        scores=torch.empty(0, device=boxes.device),
                        classes=torch.empty(0, dtype=torch.int32, device=boxes.device))

        boxes_cat   = torch.cat(all_boxes)
        scores_cat  = torch.cat(all_scores)
        labels_cat  = torch.cat(all_labels)

        order = scores_cat.argsort(descending=True)
        boxes_cat  = boxes_cat[order]
        scores_cat = scores_cat[order]
        labels_cat = labels_cat[order]

        return dict(boxes=boxes_cat,
                    scores=scores_cat,
                    classes=labels_cat)
    
    def simple_postprocess(self, cls_logits: torch.Tensor, pred_loc: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Повертає по 1 боксу на кожен клас (найвищий score).
        Працює з формами:
        - cls_logits: (N, A, C) або (A, C)
        - pred_loc  : (N, A, 4) або (A, 4)
        """

        # ==== 1) Приводимо форми до (A, C) і (A, 4) ====
        if cls_logits.ndim == 3:
            # Очікуємо N=1; якщо більше — візьмемо перший елемент (або прибери .squeeze у виклику)
            cls_logits = cls_logits[0]
        elif cls_logits.ndim != 2:
            raise ValueError(f"cls_logits must be (A,C) or (N,A,C), got {tuple(cls_logits.shape)}")

        if pred_loc.ndim == 3:
            pred_loc = pred_loc[0]
        elif pred_loc.ndim != 2:
            raise ValueError(f"pred_loc must be (A,4) or (N,A,4), got {tuple(pred_loc.shape)}")

        # ==== 2) Отримуємо scores по класах ====
        # ВАРІАНТ A: якщо в тебе softmax з background (клас 0 = BG):
        use_softmax_with_bg = True  # <-- ПОСТАВИ False, якщо перейшов на sigmoid без BG
        if use_softmax_with_bg:
            scores_full = F.softmax(cls_logits, dim=-1)   # (A, C)
            if scores_full.size(1) < 2:
                raise ValueError("Expect at least 2 classes (background + 1).")
            scores = scores_full[:, 1:]                   # прибираємо background -> (A, C-1)
            label_offset = 1
        else:
            # ВАРІАНТ B: sigmoid без background
            scores = cls_logits.sigmoid()                 # (A, C) — усі C це об’єктні класи
            label_offset = 0

        # ==== 3) Декодуємо бокси ====
        variances = self.anchors.variances
        boxes = Anchors.decode_boxes(pred_loc, self.anchors.center_anchors,
                                    variances[:2], variances[2:])         # (A, 4) у center
        boxes = Anchors.center_to_corner(boxes).clamp(0, 1)                 # (A, 4) у corner

        # ==== 4) По одному найкращому боксу на кожен клас ====
        # Для кожного класу знаходимо anchor з найбільшим score.
        # scores: (A, C_eff) -> беремо max по A
        best_scores, best_anchor_idx_per_class = scores.max(dim=0)          # (C_eff,), (C_eff,)
        keep = best_scores > self.conf_thresh
        if not keep.any():
            empty = torch.empty((0, 4), device=boxes.device)
            return dict(
                boxes=empty,
                scores=torch.empty(0, device=boxes.device),
                classes=torch.empty(0, dtype=torch.int32, device=boxes.device),
            )

        chosen_boxes  = boxes[best_anchor_idx_per_class[keep]]              # (K, 4)
        chosen_scores = best_scores[keep]                                   # (K,)
        chosen_labels = torch.arange(scores.size(1), device=boxes.device, dtype=torch.int32)[keep] + label_offset  # (K,)

        return dict(boxes=chosen_boxes, scores=chosen_scores, classes=chosen_labels)
