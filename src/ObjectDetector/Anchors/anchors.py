import math
from typing import List
import collections
import itertools

import torch
import torch.nn.functional as F

AnchorSizeRange = collections.namedtuple('AnchorSizeRange', ['min', 'max'])
AnchorSpec = collections.namedtuple('AnchorSpec', ['map_dim', 'stride', 'size_range', 'aspect_ratios'])

class Anchors:
    def __init__(self,
                 specs: List[AnchorSpec],
                 img_size: int, 
                 variances: List[float],
                 device: torch.device | str | None = None):
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.variances = torch.tensor(variances, dtype=torch.float32, device=self.device)
        self.aspects = [spec.aspect_ratios for spec in specs]

        self.center_anchors = self._generate_all(specs, img_size).to(device)
        self.corner_anchors = Anchors.center_to_corner(self.center_anchors).to(device)

    def _generate_all(self, specs: List[AnchorSpec], img_size: int) -> torch.Tensor:
        anchors = []
        for spec in specs:
            scale = img_size / spec.stride
            for j, i in itertools.product(range(spec.map_dim), repeat=2):
                cx = (i + 0.5) / scale
                cy = (j + 0.5) / scale

                base_size = spec.size_range.min
                w = h = base_size / img_size
                anchors.append([cx, cy, w, h])

                hybrid_size = math.sqrt(spec.size_range.min * spec.size_range.max)
                w = h = hybrid_size / img_size
                anchors.append([cx, cy, w, h])
                
                base_size = spec.size_range.min
                w = h = base_size / img_size
                
                for ratio in spec.aspect_ratios:
                    r = math.sqrt(ratio)
                    anchors.append([cx, cy, w * r, h / r])
                    anchors.append([cx, cy, w / r, h * r])

        anchor_tensor = torch.tensor(anchors)
        torch.clamp(anchor_tensor, 0.0, 1.0, out=anchor_tensor)
        return anchor_tensor
    
    @staticmethod
    def encode_boxes(boxes: torch.Tensor, anchor_boxes: torch.Tensor, center_var: List[float], size_var: List[float]):
        if anchor_boxes.dim() + 1 == boxes.dim():
            anchor_boxes = anchor_boxes.unsqueeze(0)
        return torch.cat([
            (boxes[..., :2] - anchor_boxes[..., :2]) / anchor_boxes[..., 2:] / center_var,
            torch.log(boxes[..., 2:] / anchor_boxes[..., 2:]) / size_var
        ], dim=boxes.dim() - 1)
    
    @staticmethod
    def decode_boxes(locations: torch.Tensor, center_anchors: torch.Tensor, center_var: List[float], size_var: List[float]):
        if center_anchors.dim() + 1 == locations.dim():
            center_anchors = center_anchors.unsqueeze(0)
        return torch.cat([
            locations[..., :2] * center_var * center_anchors[..., 2:] + center_anchors[..., :2],
            torch.exp(locations[..., 2:] * size_var) * center_anchors[..., 2:]
        ], dim=locations.dim() - 1)

    @staticmethod
    def _area_of(left_top, right_bottom) -> torch.Tensor:
        hw = torch.clamp(right_bottom - left_top, min=0.0)
        return hw[..., 0] * hw[..., 1]

    @staticmethod
    def _iou(boxes0, boxes1, eps=1e-5):
        overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = Anchors._area_of(overlap_left_top, overlap_right_bottom)
        area0 = Anchors._area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = Anchors._area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)


    def assign_priors(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, iou_thr: float = 0.5):
        iou = self._iou(gt_boxes.unsqueeze(0), self.corner_anchors.unsqueeze(1))
        best_target, best_idx = iou.max(1)
        _, best_iou = iou.max(0)

        for target_index, prior_index in enumerate(best_iou):
            best_idx[prior_index] = target_index

        best_target.index_fill(0, best_iou, 2)
        labels = gt_labels[best_idx]
        labels[best_target < iou_thr] = 0
        
        boxes = gt_boxes[best_idx]

        return boxes, labels
    
    def match(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, iou_thr: float = 0.5):
        boxes, labels = self.assign_priors(gt_boxes, gt_labels, iou_thr)
        boxes = self.corner_to_center(boxes)
        deltas = self.encode_boxes(boxes, self.center_anchors, self.variances[:2], self.variances[2:])
        return deltas, labels
    
    @staticmethod
    def center_to_corner(boxes: torch.Tensor):
        return torch.cat([boxes[..., :2] - boxes[..., 2:] / 2,
                            boxes[..., :2] + boxes[..., 2:] / 2], boxes.dim() - 1)

    @staticmethod
    def corner_to_center(boxes: torch.Tensor):
        return torch.cat([
            (boxes[..., :2] + boxes[..., 2:]) / 2,
            boxes[..., 2:] - boxes[..., :2]
        ], boxes.dim() - 1)
        