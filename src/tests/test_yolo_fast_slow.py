import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from ObjectDetector.Yolo.Models.yolo_fast_and_slow import YoloFastAndSlow

B, T, H, W = 2, 3, 640, 640
device = "cuda" if torch.cuda.is_available() else "cpu"

labels = [""] * 80
model = YoloFastAndSlow(
    labels,
    "Model/yolo11n.pt",
    "Model/yolo11x.pt",
    device=device
)

x_seq = torch.randn(B, T, 3, H, W, device=device)

model.eval()
logits_seq = model.forward(x_seq)

assert isinstance(logits_seq, list), "Model in eval mode must return list"
assert len(logits_seq) == T, f"Expected T={T} outputs, got {len(logits_seq)}"

assert isinstance(logits_seq[0], list), "Each timestep output must be a list (batch)"
assert len(logits_seq[0]) == B, f"Batch size must be {B}, got {len(logits_seq[0])}"

first_box = logits_seq[0][0]
assert hasattr(first_box, "boxes"), "YOLO prediction must have .boxes attribute"

xyxy_shape = first_box.boxes.xyxyn.shape
assert xyxy_shape[-1] == 4, f"xyxy must have shape (_,4), got {xyxy_shape}"

cls_tensor = first_box.boxes.cls
assert cls_tensor.ndim == 1, "class vector must be 1D"

model.train()
logits_seq_train = model.forward(x_seq)

assert len(logits_seq_train) == T, "Train mode must return sequence of length T"
assert isinstance(logits_seq_train[0], list), "Train mode must return list per timestep"
assert logits_seq_train[0][0].shape[0] > 0, "Train logits must have non-zero predictions"

model.eval()
out1 = model.forward(x_seq)
out2 = model.forward(x_seq)

b1 = out1[0][0].boxes.xyxyn
b2 = out2[0][0].boxes.xyxyn
assert torch.allclose(b1, b2, atol=1e-4), "Eval mode should be deterministic"

model.train()
out3 = model.forward(x_seq)
assert isinstance(out3[0][0], torch.Tensor), "Train output must be raw tensor predictions"
assert out3[0][0].shape[-1] > 4, "Train prediction tensor must contain >4 values per bbox (cls, obj, etc.)"

assert next(model.parameters()).device.type == device, "Model must be on selected device"

print("✔ All tests passed!")