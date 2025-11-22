import torch
import os
from Dataset.voc_dataset import VOCDataset
from ObjectDetector.Yolo.Models.yolo_fast_and_slow import YoloFastAndSlow
from ObjectDetector.Shared.Models.conv_lstm import MultiScaleConvLSTM, Adapter, ConvLSTMCell, Conv2dLN

net = torch.load("Model/Yolo/fast_slow_improved_fixed_2.pt", weights_only=False)

model = YoloFastAndSlow(VOCDataset.VOC_CLASSES, "Model/Yolo/yolo11n_voc.pt", "Model/Yolo/yolo11x_voc.pt")
model.load_state_dict(net["state_dict"])

path = "Model/Yolo/fast_slow_improved_fixed.pt"
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
torch.save(model.state_dict(), path)
