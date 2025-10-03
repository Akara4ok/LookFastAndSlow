import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from ObjectDetector.Models.interleaved_classifier import InterleavedClassifier

B,T,H,W = 8,3,300,300
num_classes = 100
lstm_channels = [576, 1280, 512, 256, 256, 64]

model = InterleavedClassifier(300, 0.5, 1.0, lstm_channels, num_classes)

# fake input
x_seq = torch.randn(B,T,3,H,W)
y = torch.randint(0,num_classes,(B,))

# forward
logits_seq = model.forward(x_seq)   # (B,T,num_classes)
print("logits_seq:", logits_seq.shape)

# loss averaged over steps
loss = 0
for t in range(T):
    loss += F.cross_entropy(logits_seq[:,t], y)
loss = loss / T
print("loss:", loss.item())

# backward
loss.backward()
print("Backprop OK, grads nonzero:",
        any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

torch.save(model.state_dict(), "phase1.weights.h5")
