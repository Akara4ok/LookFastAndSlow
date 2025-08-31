import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dLN(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, bias=True, layernorm=False):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=bias)
        self.ln = nn.LayerNorm(out_ch) if layernorm else None

        nn.init.orthogonal_(self.conv.weight, gain=1.0)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        y = self.conv(x)
        if self.ln is not None:
            y = y.permute(0, 2, 3, 1)           # (B,H,W,C)
            y = self.ln(y)
            y = y.permute(0, 3, 1, 2).contiguous()
        return y


class ConvLSTMCell(nn.Module):
    """
    Standard ConvLSTM cell with 4 gates (i, f, o, g):
      i_t = o(Wxi * x_t + Whi * h_{t-1} + Wci ⊙ c_{t-1} + b_i)   (peephole optional)
      f_t = o(Wxf * x_t + Whf * h_{t-1} + Wcf ⊙ c_{t-1} + b_f)
      g_t = tanh(Wxg * x_t + Whg * h_{t-1} + b_g)                 (candidate)
      c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
      o_t = o(Wxo * x_t + Who * h_{t-1} + Wco ⊙ c_t + b_o)
      h_t = o_t ⊙ tanh(c_t)
    """
    def __init__(self, in_ch, hid_ch, k=3, bias=True, layernorm=False, peephole=True):
        super().__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.peephole = peephole

        # One conv for input, one for hidden; each outputs 4*hid_ch gates
        self.x2g = Conv2dLN(in_ch, 4 * hid_ch, k, bias=bias, layernorm=False)
        self.h2g = Conv2dLN(hid_ch, 4 * hid_ch, k, bias=False, layernorm=False)

        # LayerNorm on each gate block optionally (over channels)
        self.ln = nn.LayerNorm(4 * hid_ch) if layernorm else None

        if self.peephole:
            # separate learnable peephole weights per gate that uses c
            self.w_ci = nn.Parameter(torch.zeros(1, hid_ch, 1, 1))
            self.w_cf = nn.Parameter(torch.zeros(1, hid_ch, 1, 1))
            self.w_co = nn.Parameter(torch.zeros(1, hid_ch, 1, 1))

        # Bias init trick: forget gate bias to positive value → longer memory
        # We do this by manually splitting x2g bias (since h2g has no bias)
        if bias and self.x2g.conv.bias is not None:
            with torch.no_grad():
                # gates order: [i, f, g, o]
                b = self.x2g.conv.bias.view(4, hid_ch)
                b[1].fill_(1.0)  # forget gate bias = 1.0
                self.x2g.conv.bias.copy_(b.view(-1))

    def init_state(self, x):
        B, _, H, W = x.shape
        h = torch.zeros(B, self.hid_ch, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hid_ch, H, W, device=x.device, dtype=x.dtype)
        return h, c

    def forward(self, x_t, state=None):
        if state is None:
            h_prev, c_prev = self.init_state(x_t)
        else:
            h_prev, c_prev = state

        gates = self.x2g(x_t) + self.h2g(h_prev)  # (B, 4*hid_ch, H, W)

        if self.ln is not None:
            # (B,4C,H,W) -> (B,H,W,4C) for LN over channels, then back
            gates = gates.permute(0, 2, 3, 1)          # (B,H,W,4C)
            gates = self.ln(gates)                     # LN over last dim (4C)
            gates = gates.permute(0, 3, 1, 2).contiguous()  # (B,4C,H,W)  <-- FIXED


        # Split gates
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        if self.peephole:
            i = i + self.w_ci * c_prev
            f = f + self.w_cf * c_prev

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g

        if self.peephole:
            o = o + self.w_co * c_t

        o = torch.sigmoid(o)
        h_t = o * torch.tanh(c_t)
        return h_t, c_t
