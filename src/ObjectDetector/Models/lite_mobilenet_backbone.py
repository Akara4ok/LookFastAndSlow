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
    def __init__(self, input_size: int = 300):
        super().__init__()
        self.input_size = input_size

        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features

        self.connectors = nn.ModuleList([
            InvertedResidualBlock(1280, 512, 2, 0.20),
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
    
class MobileNetV2Phase1(nn.Module):
    def __init__(self, width_mult=1.0, out_channels=256):
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1, width_mult=width_mult)
        self.features = base.features
        self.proj = nn.Conv2d(base.last_channel, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = self.proj(f)
        return f
