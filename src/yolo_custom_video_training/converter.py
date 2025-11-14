import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ObjectDetector.Shared.Models.conv_lstm import MultiScaleConvLSTM, Adapter, ConvLSTMCell, Conv2dLN

from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.nms import non_max_suppression
from ultralytics.engine.results import Results

class YoloFastAndSlow(nn.Module):
    def __init__(self, 
                 names,
                 weights_small="yolo11n.pt",
                 weights_large="yolo11x.pt",
                 device="cuda"):
        super().__init__()
        self.device = device

        small_model = YOLO(weights_small)
        large_model = YOLO(weights_large)

        self.backbone_small = small_model.model.to(device).eval()
        self.backbone_large = large_model.model.to(device).eval()

        for p in self.backbone_small.parameters():
            p.requires_grad = False
        for p in self.backbone_large.parameters():
            p.requires_grad = False
        
        dummy = torch.zeros(1, 3, 320, 320).to(device)
        with torch.no_grad():
            feats_s = self._extract_feats(self.backbone_small, dummy)
            feats_l = self._extract_feats(self.backbone_large, dummy)

        in_chs_small = [f.shape[1] for f in feats_s]
        in_chs_large = [f.shape[1] for f in feats_l]

        ref_head = large_model.model.model[-1]
        detect_in_chs = []
        for seq in ref_head.cv2:
            conv_layers = [m for m in seq.modules() if isinstance(m, torch.nn.Conv2d)]
            if len(conv_layers) > 0:
                detect_in_chs.append(conv_layers[0].in_channels)
        
        self.adapter_small = Adapter(in_chs_small, detect_in_chs).to(device)
        self.adapter_large = Adapter(in_chs_large, detect_in_chs).to(device)

        self.temporal = MultiScaleConvLSTM(in_chs=detect_in_chs, hid_chs=detect_in_chs).to(device)

        self.detect = Detect(nc=len(names), ch=detect_in_chs).to(device)

        with torch.no_grad():
            if hasattr(ref_head, "stride"):
                self.detect.stride = ref_head.stride.clone()
            if hasattr(ref_head, "anchors"):
                self.detect.anchors = ref_head.anchors.clone()
            state = ref_head.state_dict()
            self.detect.load_state_dict(state, strict=False)

        self.names = names
        self.args = None
        self.seq = True

        self.current_t = 0
        self.state = None

    def _extract_feats(self, full_model, x):
        # full_model is e.g. small_model.model (not sliced!)
        y, outputs = None, []
        for i, m in enumerate(full_model.model):
            if m.f != -1:
                x = [x if j == -1 else outputs[j] for j in (m.f if isinstance(m.f, list) else [m.f])]
            x = m(x)
            outputs.append(x)
        # Take outputs from the same layers Detect head uses (P3,P4,P5)
        detect_layer = full_model.model[-1]
        feat_idxs = detect_layer.f
        feats = [outputs[i] for i in feat_idxs]
        return feats
    
    @property
    def model(self):
        # mimic YOLO internal structure
        return nn.ModuleList([self.detect])
    
    def loss(self, pred, batch):
        if not hasattr(self, "_loss_fn"):
            from ultralytics.utils.loss import v8DetectionLoss
            from types import SimpleNamespace
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
            self._loss_fn = v8DetectionLoss(self)

        
        was_training = self.training
        self.train()

        loss_value = self._loss_fn(pred, batch)

        if(not was_training):
            self.eval()

        return loss_value
    
    def choose_backbone_adapter(self, t: int):
        return (self.backbone_large, self.adapter_large, True) if (t % 2 == 0) else (self.backbone_small, self.adapter_small, False)
    
    def choose_cur_backbone_adapter(self):
        return (self.backbone_large, self.adapter_large, True) if (self.current_t % 1 == 0) else (self.backbone_small, self.adapter_small, False)

    @torch.no_grad()
    def postprocess(self, preds: torch.Tensor, imgs: torch.Tensor,
                    conf_thres: float = 0.25, iou_thres: float = 0.45):
        device = self.device
        bs = preds.shape[0]
        nc = preds.shape[1] - 4

        dets = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)

        results = []
        for i in range(bs):
            img = imgs[i]
            h, w = img.shape[1:]
            d = dets[i]

            if d is None or len(d) == 0:
                boxes = torch.zeros((0, 6), device=device)
            else:
                # xywh → xyxy, rescale to image size
                boxes = scale_boxes(img.shape[1:], d[:, :4], (h, w)).round()
                boxes = torch.cat([boxes, d[:, 4:6]], 1)  # [x1,y1,x2,y2,conf,cls]

            results.append(
                Results(
                    orig_img=img.permute(1, 2, 0).cpu().numpy(),
                    path=None,
                    boxes=boxes,
                    names=self.names,
                )
            )
        return results

    def forward(self, frames: torch.Tensor):
        if(self.seq):
            return self.forward_seq(frames)
        else:
            return self.forward_frame(frames)
        
    def predict(self, frames: torch.Tensor):
        return self.forward(frames)

    def forward_frame(self, frames: torch.Tensor):
        self.eval()

        x_t = frames
        backbone, adapter, is_large = self.choose_cur_backbone_adapter()

        with torch.no_grad():
            feats = self._extract_feats(backbone, x_t)

        adapted = adapter(feats)

        if self.state is None:
            self.state = self.temporal.init_states(adapted)

        out_feats, state = self.temporal.step(adapted, self.state)
        if(is_large):
            self.state = state
        
        y = self.detect.forward(out_feats)
        
        self.current_t += 1

        return self.postprocess(y[0], x_t)

    def forward_seq(self, frames: torch.Tensor):
        B, T, C, H, W = frames.shape
        outputs = []
        states = None

        for t in range(T):
            x_t = frames[:, t]
            backbone, adapter, is_large = self.choose_backbone_adapter(t)

            with torch.no_grad():
                feats = self._extract_feats(backbone, x_t)

            adapted = adapter(feats)

            if states is None:
                states = self.temporal.init_states(adapted)

            out_feats, states_after = self.temporal.step(adapted, states)
            if(is_large):
                states = states_after
                
            y = self.detect.forward(out_feats)

            if(self.detect.training):
                outputs.append(y)
            else:
                outputs.append(self.postprocess(y[0], x_t))

        return outputs

full_model = torch.load("Model/Yolo/fast_slow_2.pt", map_location="cuda", weights_only=False)
full_model = full_model["model"]
full_model.eval().to("cuda")

from ObjectDetector.Yolo.Models.yolo_fast_and_slow import YoloFastAndSlow

new_model = YoloFastAndSlow(full_model.names, "Model/Yolo/yolo11n_voc.pt", "Model/Yolo/yolo11x_voc.pt")
new_model.backbone_small = full_model.backbone_small
new_model.backbone_large = full_model.backbone_large
new_model.adapter_small = full_model.adapter_small
new_model.adapter_large = full_model.adapter_large
new_model.temporal = full_model.temporal
new_model.detect = full_model.detect

torch.save({"model": new_model, "state_dict": new_model.state_dict()}, "Model/Yolo/fast_slow_converted.pt")
