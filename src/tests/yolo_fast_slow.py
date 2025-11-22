import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from ObjectDetector.Yolo.Models.yolo_fast_and_slow import YoloFastAndSlow
from Dataset.voc_dataset import VOCDataset

B,T,H,W = 4,6,640,640
num_classes = 100
lstm_channels = [64, 128, 512]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YoloFastAndSlow(VOCDataset.VOC_CLASSES, "Model/yolo11n.pt", "Model/yolo11x.pt", False, device=device)

x_seq = torch.randn(B,T,3,H,W, device=device)

print("Eval")
model.eval()
logits_seq = model.forward(x_seq)
print("logits_seq len:", len(logits_seq))
print("logits_seq batch:", len(logits_seq[0]))
print("logits_seq:", logits_seq[0][0].boxes.xyxy.shape)
print("logits_seq:", logits_seq[0][0].boxes.cls)

print("Train")
model.train()
logits_seq = model.forward(x_seq)
print("logits_seq len:", len(logits_seq))
print("logits_seq batch len:", len(logits_seq[0]))
print("logits_seq batch: shape", logits_seq[0][0].shape)