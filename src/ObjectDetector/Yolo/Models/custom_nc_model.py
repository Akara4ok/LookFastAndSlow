import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics import YOLO

class CustomDetect(Detect):
    def __init__(self, nc: int, ch: tuple = ()):
        super().__init__(nc=nc, ch=ch)
        self.f = [16, 19, 22]

    def forward(self, x):
        # якщо нам передали не список, а один тензор -> це neck-вихід; треба взяти фічі з self.f
        if not isinstance(x, (list, tuple)):
            # Ultralytics у predict/train проході тримає всі попередні виходи в self._forward_once
            # але при побудові stride'ів нам приходить "теплий" виклик з фейковим тензором
            # тому просто обертаємо його у список трьох копій потрібної форми
            x = [x, x, x]
        elif isinstance(x[0], torch.Tensor) and x[0].dim() == 3:
            # якщо випадково прилетіли 3D-тензори, додай просторову ось
            x = [t.unsqueeze(-1) for t in x]

        return super().forward(x)

# class CustomDetect(Detect):
#     def __init__(self, nc: int, ch: tuple, f: list,  i: list):
#         super().__init__(nc=nc, ch=ch)
#         self.f = f if f is not None else [-1]
#         self.i = i if i is not None else -1

def build_test_model(labels = list, pretrained_backbone_weights: str = "yolo11x.pt"):
    base = YOLO(pretrained_backbone_weights)
    mdl = base.model

    detect_layer = None
    for m in mdl.model[::-1]:  # йдемо з кінця
        if isinstance(m, Detect):
            detect_layer = m
            break
    if detect_layer is None:
        raise RuntimeError("Detect layer not found in model (YOLOv11 architecture changed).")
    
    stride = getattr(mdl, "stride", [8, 16, 32])
    if isinstance(stride, torch.Tensor):
        stride = stride.tolist()
    reg_max = getattr(detect_layer, "reg_max", 16)

    # -----------------------------
    # дістати канали з detect.cv3
    # -----------------------------
    ch = []
    for i, seq in enumerate(detect_layer.cv3):
        # знайти перший Conv2d з in_channels
        found = None
        for sub in seq.modules():
            if isinstance(sub, torch.nn.Conv2d):
                found = sub.in_channels
                break
        if found is None:
            raise RuntimeError(f"Cannot extract in_channels for branch {i}")
        ch.append(found)

    print(f"[INFO] Found ch={ch}, stride={stride}, reg_max={reg_max}")

    # -----------------------------
    # створюємо нову голову
    # -----------------------------
    nc = len(labels)
    f = getattr(detect_layer, "f", [-1])
    i = getattr(detect_layer, "i", -1)
    new_head = CustomDetect(nc=nc, ch=ch, f=f, i=i)

    # замінюємо стару голову
    mdl.model[-1] = new_head
    mdl.names = {i: name for i, name in enumerate(labels)}

    return mdl



if __name__ == "__main__":
    # model = YOLO("Model/yolo11x_test.pt")

    labels = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]
    model = build_test_model(labels, "Model/yolo11x.pt")

    torch.save({"model": model, "state_dict": model.state_dict()}, "Model/yolo11x_test.pt")
    print("Saved yolo11x_test.pt")



    # from ultralytics import YOLO
    # m = YOLO("Model/yolo11x.pt").model

    # print(m)                     # короткий опис архітектури
    # print("\n--- attrs ---")
    # for name in dir(m.model[-1]):
    #     if not name.startswith("_"):
    #         print(name)
