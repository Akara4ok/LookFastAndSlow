import random
import torch
import torch.nn as nn

from ObjectDetector.Models.lite_mobilenet_backbone import LiteMobileNetBackbone, _infer_pyramid
from ObjectDetector.Models.conv_lstm import Adapter, MultiScaleConvLSTM

class ClassificationHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.gap(x).flatten(1)
        return self.fc(x)
    
class InterleavedClassifier(nn.Module):
    def __init__(self, img_size: int, fast_width: float, slow_width: float, 
                 lstm_chs: list[int], num_classes: int):
        super().__init__()
        self.fast = LiteMobileNetBackbone(input_size=img_size, width_mult=fast_width)
        self.slow = LiteMobileNetBackbone(input_size=img_size, width_mult=slow_width)
        
        device = next(self.slow.parameters()).device if any(p.requires_grad for p in self.slow.parameters()) else "cpu"
        fast_chs, _ = _infer_pyramid(self.fast, img_size, device)
        slow_chs, _ = _infer_pyramid(self.slow, img_size, device)
        
        self.fast_adapter = Adapter(fast_chs, lstm_chs)
        self.slow_adapter = Adapter(slow_chs, lstm_chs)
        
        self.mslstm = MultiScaleConvLSTM(in_chs=lstm_chs, hid_chs=lstm_chs, k=3)
        
        self.sum_hid = sum(lstm_chs)
        self.head = ClassificationHead(self.sum_hid, num_classes)
        
    def _head_from_hidden(self, h_feats: list[torch.Tensor]) -> torch.Tensor:
        pooled = [h.mean(dim=(2, 3), keepdim=True) for h in h_feats]
        x = torch.cat(pooled, dim=1)
        return x

    def forward(self, x_seq: torch.Tensor, state: list = None):
        B, T, C, H, W = x_seq.shape

        logits_list = []
        states = state

        for t in range(T):
            x_t = x_seq[:, t]

            if random.random() < 0.5:
                feats = self.fast(x_t)
                feats = self.fast_adapter(feats)
            else:
                feats = self.slow(x_t)
                feats = self.slow_adapter(feats)

            if states is None:
                states = self.mslstm.init_states(feats)

            h_feats, states = self.mslstm.step(feats, states)

            head_in = self._head_from_hidden(h_feats)
            logits = self.head(head_in)
            logits_list.append(logits)

        return torch.stack(logits_list, dim=1)
