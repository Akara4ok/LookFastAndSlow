from ultralytics import YOLO
import numpy as np
import torch
import time

# model = YOLO("Model/Yolo/yolo11x_voc.pt")
model = YOLO("src/ObjectDetector/Yolo/Models/custom_head.yaml", task = "detect")
model.load("Model/Yolo/yolo11x_voc.pt")
model_pt = model.model.to("cuda")

layers_to_hook = [16, 19, 22]   # наші виходи
features = {}

# define hook
def get_hook(name):
    def hook(module, input, output):
        features[name] = output
    return hook

# register hooks
hooks = []
# for idx in layers_to_hook:
    # h = model_pt.model[idx].register_forward_hook(get_hook(idx))
    # hooks.append(h)

# run dummy
x = torch.zeros(1, 3, 640, 640, device="cuda")

for i in range(50):
    with torch.no_grad():
        _ = model_pt(x)

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    _ = model_pt(x)
torch.cuda.synchronize()

print("Time", time.time() - t0)

# print results
for k, v in features.items():
    print(f"Layer {k}: {v.shape}")

# remove hooks (важливо!)
# for h in hooks:
    # h.remove()
