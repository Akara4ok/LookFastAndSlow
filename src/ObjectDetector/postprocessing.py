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
                        classes=torch.empty(0, dtype=torch.int32, device=boxes.device),
                        num_detections=torch.tensor([0], dtype=torch.int32))

        boxes_cat   = torch.cat(all_boxes)
        scores_cat  = torch.cat(all_scores)
        labels_cat  = torch.cat(all_labels)

        order = scores_cat.argsort(descending=True)[: self.top_k]
        boxes_cat  = boxes_cat[order]
        scores_cat = scores_cat[order]
        labels_cat = labels_cat[order]

        return dict(boxes=boxes_cat,
                    scores=scores_cat,
                    classes=labels_cat,
                    num_detections=torch.tensor([boxes_cat.size(0)],
                                                dtype=torch.int32,
                                                device=boxes_cat.device))