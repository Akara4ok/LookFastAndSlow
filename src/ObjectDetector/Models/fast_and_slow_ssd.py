import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from ObjectDetector.Models.conv_lstm import MultiScaleConvLSTM, Adapter
from ObjectDetector.Models.ssd_head import SSDHead 
from ObjectDetector.Models.lite_mobilenet_backbone import LiteMobileNetBackbone 

def _dummy_infer(backbone: nn.Module, img_size: int = 300, device: str = "cpu"):
    backbone = backbone.to(device)
    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size, device=device)
        feats: List[torch.Tensor] = backbone(x)  # list of maps
    chs = [f.shape[1] for f in feats]
    shapes = [(f.shape[2], f.shape[3]) for f in feats]  # (H,W)
    return chs, shapes

class LookFastSlowSSD(nn.Module):
    def __init__(self,
                 num_classes: int,
                 aspects: List[List[float]],
                 img_size: int = 300,
                 fast_width: float = 0.5,
                 lstm_kernel: int = 3,
                 run_slow_every: int = None
                 ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.aspects = aspects
        self.run_slow_every = run_slow_every

        lstm_channels = [576, 1280, 512, 256, 256, 64]

        self.slow_extractor = LiteMobileNetBackbone(input_size=img_size)
        self.fast_extractor = LiteMobileNetBackbone(input_size=img_size, width_mult=fast_width)

        device = next(self.slow_extractor.parameters()).device if any(p.requires_grad for p in self.slow_extractor.parameters()) else "cpu"
        slow_chs, slow_shapes = _dummy_infer(self.slow_extractor, img_size=img_size, device=device)
        fast_chs, fast_shapes = _dummy_infer(self.fast_extractor, img_size=img_size, device=device)

        self.adapt_fast = Adapter(fast_chs, lstm_channels)
        self.adapt_slow = Adapter(slow_chs, lstm_channels)

        self.mslstm = MultiScaleConvLSTM(in_chs=lstm_channels, hid_chs=lstm_channels,
                                         k=lstm_kernel)

        self.ssd_head = SSDHead(out_channels=lstm_channels, num_classes=num_classes, aspects=aspects)

    def _make_schedule(self, T: int) -> List[bool]:
        if self.training:
            return [bool(torch.randint(0, 2, ()).item()) for _ in range(T)]
        else:
            k = self.run_slow_every
            return [(t % k == 0) for t in range(T)]

    def forward(self, x_seq: torch.Tensor):
        B, T, C, H, W = x_seq.shape
        assert H == self.img_size and W == self.img_size, "Resize your inputs to img_size before calling."

        use_slow = self._make_schedule(T)

        locs_steps, confs_steps = [], []
        states = None

        for t in range(T):
            x_t = x_seq[:, t]  # (B,3,H,W)

            if use_slow[t]:
                feats: List[torch.Tensor] = self.slow_extractor(x_t)
                feats = self.adapt_slow.forward(feats)
            else:
                feats: List[torch.Tensor] = self.fast_extractor(x_t)
                feats = self.adapt_fast.forward(feats)

            if states is None:
                states = self.mslstm.init_states(feats)
            h_feats, states = self.mslstm.step(feats, states)

            locs_t, confs_t = self.ssd_head(h_feats)
            locs_steps.append(locs_t)
            confs_steps.append(confs_t)

        locs_all  = torch.stack(locs_steps,  dim=1)
        confs_all = torch.stack(confs_steps, dim=1)
        used_mask = torch.tensor(use_slow, device=locs_all.device, dtype=torch.bool)
        return locs_all, confs_all, used_mask
    