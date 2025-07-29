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
    num_neg = num_pos * neg_pos_ratio

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
        
        num_classes = pred_logits.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(pred_logits, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)
        
        pred_logits = pred_logits[mask, :]
        
        cls_loss = F.cross_entropy(pred_logits.reshape(-1, num_classes), gt_labels[mask], reduction='sum')
        pos_mask = gt_labels > 0
        pred_deltas = pred_deltas[pos_mask, :].reshape(-1, 4)
        gt_deltas = gt_deltas[pos_mask, :].reshape(-1, 4)
        
        loc_loss = F.smooth_l1_loss(pred_deltas, gt_deltas, reduction='sum')
        num_pos = max(gt_deltas.size(0), 1)

        return loc_loss/num_pos, cls_loss/num_pos