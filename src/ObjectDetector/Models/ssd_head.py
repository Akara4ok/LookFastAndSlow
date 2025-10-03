import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """
    Depth-wise separable conv mirroring tf.keras.layers.SeparableConv2D.
    """
    def __init__(self, cin, cout, k=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, stride, padding,
                            groups=cin, bias=bias)
        self.pw = nn.Conv2d(cin, cout, 1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


class SSDHead(nn.Module):
    def __init__(self,
                 out_channels,
                 num_classes: int,
                 aspects):
        super().__init__()
        self.num_classes = num_classes
        aspect_sizes = [2 * len(a) + 2 for a in aspects]      # +1 for 1:1 default

        cls_layers, box_layers = [], []
        for idx, (out_channel, a_size) in enumerate(zip(out_channels, aspect_sizes)):
            # All maps except the last use 3×3 separable conv
            if idx < len(out_channels) - 1:
                cls_layers.append(
                    SeparableConv2d(out_channel, a_size * num_classes)
                )
                box_layers.append(
                    SeparableConv2d(out_channel, a_size * 4)
                )
            else:  # final map – 1×1 convs
                cls_layers.append(nn.Conv2d(out_channel,
                                            a_size * num_classes, 1))
                box_layers.append(nn.Conv2d(out_channel,
                                            a_size * 4, 1))
        self.cls_convs = nn.ModuleList(cls_layers)
        self.box_convs = nn.ModuleList(box_layers)

    @staticmethod
    def _permute(x):
        return x.permute(0, 2, 3, 1).contiguous()   # (N,C,H,W) -> (N,H,W,C)

    @staticmethod
    def _flatten(x, dim):
        n, h, w, c = x.shape
        return x.view(n, -1, dim)                   # (N,H*W*anchors,dim)

    def forward(self, features):
        locs, confs = [], []
        for feat, cls_conv, box_conv in zip(features,
                                            self.cls_convs,
                                            self.box_convs):
            cls = self._flatten(self._permute(cls_conv(feat)),
                                 self.num_classes)
            box = self._flatten(self._permute(box_conv(feat)), 4)
            confs.append(cls)
            locs.append(box)

        confs = torch.cat(confs, dim=1)             # (N, ΣA, num_classes)
        locs  = torch.cat(locs,  dim=1)             # (N, ΣA, 4)
        return locs, confs