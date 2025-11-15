import torch
import torch.nn as nn

class Conv2dLN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, bias: bool = True):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=bias)

        nn.init.orthogonal_(self.conv.weight, gain=1.0)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
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
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3, bias: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch

        # One conv for input, one for hidden; each outputs 4*hid_ch gates
        self.x2g = Conv2dLN(in_ch, 4 * hid_ch, k, bias=bias)
        self.h2g = Conv2dLN(hid_ch, 4 * hid_ch, k, bias=False)

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

    def init_state(self, x: torch.Tensor) -> tuple:
        B, _, H, W = x.shape
        h = torch.zeros(B, self.hid_ch, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hid_ch, H, W, device=x.device, dtype=x.dtype)
        return h, c

    def forward(self, x_t: torch.Tensor, state: tuple = None) -> torch.Tensor:
        if state is None:
            h_prev, c_prev = self.init_state(x_t)
        else:
            h_prev, c_prev = state

        gates = self.x2g(x_t) + self.h2g(h_prev)  # (B, 4*hid_ch, H, W)

        # Split gates
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        i = i + self.w_ci * c_prev
        f = f + self.w_cf * c_prev

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g

        o = o + self.w_co * c_t

        o = torch.sigmoid(o)
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

class ConvLSTM(nn.Module):
    """Unroll a single-layer ConvLSTMCell over time."""
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3, bias: int = True):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid_ch, k, bias)

    def forward(self, x_seq: torch.Tensor, state: tuple = None, batch_first: bool = True) -> torch.Tensor:
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

class Adapter(nn.Module):
    def __init__(self, in_chs: list[int], out_chs: list[int]):
        super().__init__()
        layers = []
        for ci, co in zip(in_chs, out_chs):
            layers.append(nn.Conv2d(ci, co, kernel_size=1, bias=False))
        self.adapters = nn.ModuleList(layers)
        
    def forward(self, feats: torch.Tensor) -> list:
        return [ad(f) for ad, f in zip(self.adapters, feats)]
    
class MultiScaleConvLSTM(nn.Module):
    def __init__(self, in_chs: list[int], hid_chs: list[int], k: int = 3, use_levels=None):
        super().__init__()
        assert len(in_chs) == len(hid_chs)
        self.levels = len(in_chs)

        if use_levels is None:
            use_levels = [False] * (self.levels - 1) + [True]
        assert len(use_levels) == self.levels
        self.use_levels = use_levels

        cells = []
        for in_c, h_c, use in zip(in_chs, hid_chs, use_levels):
            if use:
                cells.append(ConvLSTMCell(in_ch=in_c, hid_ch=h_c, k=k))
            else:
                cells.append(None)  # цей рівень не має LSTM
        self.cells = nn.ModuleList([c for c in cells if c is not None])
        self.level_to_cell_idx = []
        cur = 0
        for use in use_levels:
            if use:
                self.level_to_cell_idx.append(cur)
                cur += 1
            else:
                self.level_to_cell_idx.append(None)

    def init_states(self, x_feats: list[torch.Tensor]):
        states = []
        for lvl, x in enumerate(x_feats):
            cell_idx = self.level_to_cell_idx[lvl]
            if cell_idx is None:
                states.append(None)
            else:
                cell = self.cells[cell_idx]
                states.append(cell.init_state(x))
        return states

    def step(self, x_feats: list[torch.Tensor], states: list[tuple | None]):
        outs, new_states = [], []
        for lvl, (x, st) in enumerate(zip(x_feats, states)):
            cell_idx = self.level_to_cell_idx[lvl]
            if cell_idx is None:
                outs.append(x)
                new_states.append(None)
            else:
                cell = self.cells[cell_idx]
                h, c = cell(x, st)
                outs.append(h)
                new_states.append((h, c))
        return outs, new_states
    
class MultiScaleConvLSTM2(nn.Module):
    def __init__(self, in_chs: list[int], hid_chs: list[int], k: int = 3):
        super().__init__()
        assert len(in_chs) == len(hid_chs)
        self.levels = len(in_chs)
        self.cells = nn.ModuleList([
            ConvLSTMCell(in_ch=in_c, hid_ch=h_ch, k=k)
            for in_c, h_ch in zip(in_chs, hid_chs)
        ])

    def init_states(self, x_feats: list[torch.Tensor]):
        states = []
        for cell, x in zip(self.cells, x_feats):
            states.append(cell.init_state(x))
        return states

    def step(self, x_feats: list[torch.Tensor], states: list[tuple]) -> tuple:
        outs, new_states = [], []
        for x, cell, st in zip(x_feats, self.cells, states):
            h, c = cell(x, st)
            outs.append(h)
            new_states.append((h, c))
        return outs, new_states