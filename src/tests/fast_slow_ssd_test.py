import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from ObjectDetector.Models.fast_and_slow_ssd import LookFastSlowSSD

aspects = [
    [2, 3],
    [2, 3],
    [2, 3],
    [2, 3],
    [2],
    [2],
]

# 3) Build model
model = LookFastSlowSSD(
    num_classes= 10,
    aspects=aspects,
    img_size=300,
    lstm_channels=[256, 256, 256, 192, 128, 96],
    fast_width=0.5,
    lstm_kernel=3,
    run_slow_every=6
)

B, T = 2, 6
dummy = torch.randn(B, T, 3, 300, 300)
model.train()
locs, confs, mask = model(dummy)  # random fast/slow schedule per step in training

print(locs.shape, confs.shape, mask)  # (B,T,ΣA,4), (B,T,ΣA,C)