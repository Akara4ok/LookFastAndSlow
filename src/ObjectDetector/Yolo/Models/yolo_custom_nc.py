import math
import logging

import torch.nn as nn

import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect


def create_yolo_with_custom_nc(weights_path, labels, map_classes, device):
    def set_num_classes_yolo11(model_module: nn.Module, class_names):
        new_nc = len(class_names)

        head = None
        for m in model_module.modules():
            if isinstance(m, Detect):
                head = m

        head.nc = new_nc

        def _swap_last_conv_to_nc(seq: nn.Sequential, nc: int):
            last = seq[-1]
            new = nn.Conv2d(last.in_channels, nc, kernel_size=1, bias=True)
            nn.init.zeros_(new.weight)
            nn.init.constant_(new.bias, -math.log((1 - 0.01) / 0.01))
            seq[-1] = new

        for i, seq in enumerate(head.cv3):
            _swap_last_conv_to_nc(seq, new_nc)

        if hasattr(head, "reg_max"):
            head.no = head.nc + 4 * head.reg_max

        if hasattr(head, "bias_init"):
            head.bias_init()

        (getattr(model_module, "model", model_module)).names = list(class_names)

        return model_module
    
    model = YOLO(weights_path)

    if labels is not None and len(labels) != len(model.names) and map_classes is None:
        model = set_num_classes_yolo11(model, labels)
    
    model.to(device)

    logging.info(f"Model loaded to {device} from {weights_path}")
    
    return model
    