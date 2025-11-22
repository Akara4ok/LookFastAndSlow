import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from ObjectDetector.Shared.Models.conv_lstm import ConvLSTM, ConvLSTMCell

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# synthetic input
B, T, C_in, H, W = 2, 5, 8, 15, 20
HID = 16
x_seq = torch.randn(B, T, C_in, H, W, device=device)

# 1) Step API: manual unroll
cell = ConvLSTMCell(C_in, HID, k=3).to(device)
opt1 = torch.optim.Adam(cell.parameters(), lr=1e-3)

h, c = (None, None)
hs = []
for t in range(T):
    if(h is None and c is None):
        h, c = cell(x_seq[:, t], None)
    else:
        h, c = cell(x_seq[:, t], (h, c))
    hs.append(h)
y_manual = torch.stack(hs, dim=1)  # (B,T,HID,H,W)

# shape checks
assert y_manual.shape == (B, T, HID, H, W), "manual unroll shape mismatch"

# 2) Wrapper API: should match manual when weights are the same
wrap = ConvLSTM(C_in, HID, k=3).to(device)
wrap.cell.load_state_dict(cell.state_dict())  # sync weights

y_wrap, (hT, cT) = wrap(x_seq, batch_first=True)
assert y_wrap.shape == (B, T, HID, H, W), "wrapper shape mismatch"

# numerical closeness
max_abs_diff = (y_wrap.detach() - y_manual.detach()).abs().max().item()
print(f"Max |y_wrap - y_manual| = {max_abs_diff:.6f}")
assert max_abs_diff < 1e-6, "Wrapper output does not match manual unroll."

# simple loss → backward
loss1 = y_manual.mean() + y_manual.pow(2).mean() * 0.1
opt1.zero_grad(); loss1.backward()
# param grads exist?
total_grad_norm = 0.0
for p in cell.parameters():
    if p.grad is not None:
        total_grad_norm += p.grad.norm().item()
assert total_grad_norm > 0, "No gradients flowed in ConvLSTMCell (manual)."
opt1.step()

# wrapper backward
opt2 = torch.optim.Adam(wrap.parameters(), lr=1e-3)
loss2 = (y_wrap[:, -1] ** 2).mean()  # use last step
opt2.zero_grad(); loss2.backward()
grad_norm2 = sum(p.grad.norm().item() for p in wrap.parameters() if p.grad is not None)
assert grad_norm2 > 0, "No gradients flowed in ConvLSTM (wrapper)."
opt2.step()

# 3) batch_first=False path
y_wrap_TB, _ = wrap(x_seq.permute(1,0,2,3,4).contiguous(), batch_first=False)
assert y_wrap_TB.shape == (T, B, HID, H, W), "batch_first=False shape mismatch"

# same values reordered?
wrap.eval()  # not strictly necessary for LN, but good hygiene
with torch.no_grad():
    # same weights, fresh state for both calls
    y_bf, _ = wrap(x_seq, batch_first=True)  # (B,T,C,H,W)

    x_tb = x_seq.permute(1,0,2,3,4).contiguous()  # (T,B,C,H,W), make contiguous
    y_tb, _ = wrap(x_tb, batch_first=False)       # (T,B,C,H,W)

    # reorder TB output to match BF layout
    y_tb_as_bf = y_tb.permute(1,0,2,3,4)          # (B,T,C,H,W)

diff_tb = (y_tb_as_bf - y_bf).abs().max().item()
print(f"Max |TB-reordered - BF| = {diff_tb:.6f}")
assert diff_tb < 1e-6

print("✅ ConvLSTMCell & ConvLSTM basic correctness checks passed.")
