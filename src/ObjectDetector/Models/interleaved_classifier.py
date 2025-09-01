import random
import torch
import torch.nn as nn

from ObjectDetector.Models.lite_mobilenet_backbone import MobileNetV2Phase1
from ObjectDetector.Models.conv_lstm import ConvLSTMCell

class ClassificationHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.gap(x).flatten(1)
        return self.fc(x)
    
class InterleavedClassifier(nn.Module):
    def __init__(self, fast_width: float, slow_width: float, backbone_out_channels: int, lstm_out_channels: int, num_classes: int):
        super().__init__()
        self.fast = MobileNetV2Phase1(fast_width, backbone_out_channels)
        self.slow = MobileNetV2Phase1(slow_width, backbone_out_channels)
        self.lstm = ConvLSTMCell(in_ch=backbone_out_channels, hid_ch=lstm_out_channels)
        self.head = ClassificationHead(lstm_out_channels, num_classes)

    def forward(self, x_seq, state=None):
        B,T,C,H,W = x_seq.shape
        logits_list = []
        h,c = state, state
        h, c = (None, None)
        for t in range(T):
            backbone = random.choice([self.fast, self.slow])
            f_t = backbone(x_seq[:,t])
            h, c = self.lstm(f_t, (h,c) if h is not None else None)
            logits = self.head(h)
            logits_list.append(logits)
        return torch.stack(logits_list, dim=1)
