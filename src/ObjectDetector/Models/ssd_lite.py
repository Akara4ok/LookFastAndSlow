import torch.nn as nn
from ObjectDetector.Models.lite_mobilenet_backbone import LiteMobileNetBackbone
from ObjectDetector.Models.ssd_head import SSDHead


class SSDLite(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_classes: int,
                 aspects: list[int]):
        super().__init__()

        self.backbone = LiteMobileNetBackbone(input_size)

        self.head = SSDHead([576, 1280, 512, 256, 256, 64],
                            num_classes,
                            aspects)

    def forward(self, x):
        feats = self.backbone(x)
        locs, confs = self.head(feats)
        return locs, confs
    