import torch
import torch.nn as nn

from ObjectDetector.Models.interleaved_classifier import InterleavedClassifier
from ObjectDetector.Models.fast_and_slow_ssd import LookFastSlowSSD

def load_phase2_from_phase1(model: LookFastSlowSSD, path: str, device: str | torch.device = "cpu"):
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
    if fast_feat_sd:
        model.fast_extractor.features.load_state_dict(fast_feat_sd, strict=False)
    if slow_feat_sd:
        model.slow_extractor.features.load_state_dict(slow_feat_sd, strict=False)

    fast_ad_sd = extract_prefix(sd, "fast_adapter.adapters.")
    slow_ad_sd = extract_prefix(sd, "slow_adapter.adapters.")
    if fast_ad_sd:
        model.adapt_fast.adapters.load_state_dict(fast_ad_sd, strict=False)
    if slow_ad_sd:
        model.adapt_slow.adapters.load_state_dict(slow_ad_sd, strict=False)

    for i, cell in enumerate(model.mslstm.cells):
        cell_sd = extract_prefix(sd, f"mslstm.cells.{i}.")
        if cell_sd:
            cell.load_state_dict(cell_sd, strict=False)

    return model