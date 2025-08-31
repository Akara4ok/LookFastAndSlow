import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_negative_mining(bg_loss: torch.Tensor,
                         labels: torch.Tensor,
                         neg_pos_ratio: int) -> torch.Tensor:
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = (num_pos * neg_pos_ratio).clamp(min=1, max=labels.size(1) - 1)

    bg_loss = bg_loss.clone()
    bg_loss[pos_mask] = -math.inf

    _, idx = bg_loss.sort(dim=1, descending=True)
    _, rank = idx.sort(dim=1)
    neg_mask = rank < num_neg
    return pos_mask | neg_mask

class SSDLoss(nn.Module):
    def __init__(self,
                 neg_pos_ratio: int = 3):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self,
                pred_deltas: torch.Tensor,      # (B, N, 4)
                pred_logits: torch.Tensor,      # (B, N, C)
                gt_deltas:  torch.Tensor,      # (B, N, 4)
                gt_labels: torch.Tensor        # (B, N)  labels
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            loss = -F.log_softmax(pred_logits, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)

        if mask.any():
            cls_loss = F.cross_entropy(pred_logits[mask], gt_labels[mask], reduction='sum')
        else:
            cls_loss = pred_logits.new_tensor(0.)

        pos_mask = gt_labels > 0
        if pos_mask.any():
            loc_loss = F.smooth_l1_loss(pred_deltas[pos_mask], gt_deltas[pos_mask], reduction='sum')
            num_pos = int(pos_mask.sum())
        else:
            loc_loss = pred_deltas.new_tensor(0.)
            num_pos  = 1

        return loc_loss/num_pos, cls_loss/num_pos
