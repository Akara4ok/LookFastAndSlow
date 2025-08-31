import torch
import torch.nn as nn
import torch.nn.functional as F4
from ObjectDetector.Models.conv_lstm_cell import ConvLSTMCell

class ConvLSTM(nn.Module):
    """Unroll a single-layer ConvLSTMCell over time."""
    def __init__(self, in_ch, hid_ch, k=3, bias=True, layernorm=False, peephole=True):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid_ch, k, bias, layernorm, peephole)

    def forward(self, x_seq, state=None, batch_first=True):
        if batch_first:
            B, T, C, H, W = x_seq.shape
            xs = [x_seq[:, t] for t in range(T)]
        else:
            T, B, C, H, W = x_seq.shape
            xs = [x_seq[t] for t in range(T)]

        outputs = []
        h, c = (state if state is not None else self.cell.init_state(xs[0]))
        for x_t in xs:
            h, c = self.cell(x_t, (h, c))
            outputs.append(h)

        if batch_first:
            y = torch.stack(outputs, dim=1)   # (B,T,Ch,H,W)
        else:
            y = torch.stack(outputs, dim=0)   # (T,B,Ch,H,W)
        return y, (h, c)
