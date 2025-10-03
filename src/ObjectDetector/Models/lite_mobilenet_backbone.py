import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import copy

class InvertedResidualBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, stride: int, expand: int):
        super().__init__()
        hidden = int(round(input_size * expand))
        self.residual = stride == 1 and input_size == output_size

        layers = []
        if(expand != 1):
            layers += [
                nn.Conv2d(input_size, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True)]
            
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(hidden, output_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_size),
            ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.residual:
            out = x + out
        return out


class LiteMobileNetBackbone(nn.Module):
    def __init__(self, input_size: int = 300, width_mult: int = 1.0):
        super().__init__()
        self.input_size = input_size

        if(width_mult == 1.0):
            mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            mobilenet = mobilenet_v2(weights=None, width_mult=width_mult)
        self.features = mobilenet.features

        self.last_channel = mobilenet.last_channel
        self.connectors = nn.ModuleList([
            InvertedResidualBlock(self.last_channel, 512, 2, 0.20),
            InvertedResidualBlock(512, 256, 2, 0.25),
            InvertedResidualBlock(256, 256, 2, 0.50),
            InvertedResidualBlock(256, 64, 2, 0.25)
        ])
        
        self.block13_expand = copy.deepcopy(nn.Sequential(
            self.features[13].conv[0],
            self.features[13].conv[1]
        ))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for i in range(13):
            x = self.features[i](x)
            
        features.append(self.block13_expand(x))
        
        x = self.features[13](x)
        # features.append(x)
        for i in range(14, len(self.features)):
            x = self.features[i](x)
        
        features.append(x)

        for extra in self.connectors:
            x = extra(x)
            features.append(x)

        return features

def _infer_pyramid(backbone: nn.Module, img_size: int, device) -> tuple:
    backbone = backbone.to(device)
    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size, device=device)
        feats: list[torch.Tensor] = backbone(x)
    chs = [f.shape[1] for f in feats]
    hw  = [(f.shape[2], f.shape[3]) for f in feats]
    return chs, hw
