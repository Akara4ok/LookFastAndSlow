import torch

from ObjectDetector.Models.interleaved_classifier import InterleavedClassifier
from ObjectDetector.Models.fast_and_slow_ssd import LookFastSlowSSD

def load_phase2_from_phase1(model: LookFastSlowSSD, path: str, device: str | torch.device = "cpu") -> LookFastSlowSSD:
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    def extract_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        plen = len(prefix)
        return {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}

    fast_feat_sd = extract_prefix(sd, "fast.features.")
    slow_feat_sd = extract_prefix(sd, "slow.features.")

    if len(fast_feat_sd):
        missing, unexpected = model.fast_extractor.features.load_state_dict(fast_feat_sd, strict=False)
        print(f"[fast.features] loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[fast.features] not found in checkpoint; skipped.")

    if len(slow_feat_sd):
        missing, unexpected = model.slow_extractor.features.load_state_dict(slow_feat_sd, strict=False)
        print(f"[slow.features] loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[slow.features] not found in checkpoint; skipped.")

    lstm_sd = extract_prefix(sd, "lstm.")
    if "x2g.conv.weight" in lstm_sd:
        in_ch1 = lstm_sd["x2g.conv.weight"].shape[1]
        hid_ch1 = lstm_sd["x2g.conv.weight"].shape[0] // 4
        copied_levels = 0
        for li, cell in enumerate(model.mslstm.cells):
            c_sd = cell.state_dict()
            in_ok  = (c_sd["x2g.conv.weight"].shape[1] == in_ch1)
            hid_ok = ((c_sd["x2g.conv.weight"].shape[0] // 4) == hid_ch1)
            if in_ok and hid_ok:
                missing, unexpected = cell.load_state_dict(lstm_sd, strict=False)
                print(f"[lstm -> level {li}] loaded: missing={len(missing)}, unexpected={len(unexpected)}")
                copied_levels += 1
            else:
                print(f"[lstm -> level {li}] shape mismatch, expected in={c_sd['x2g.conv.weight'].shape[1]},"
                      f" hid={c_sd['x2g.conv.weight'].shape[0]//4} vs phase1 in={in_ch1}, hid={hid_ch1}; skipped.")
        if copied_levels == 0:
            print("[lstm] no levels matched shapes; nothing copied.")
    else:
        print("[lstm] weights not found in checkpoint; skipped.")

    return model