import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ObjectDetector.Shared.Models.conv_lstm import MultiScaleConvLSTM, Adapter

from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.nms import non_max_suppression
from ultralytics.engine.results import Results

from ObjectDetector.profiler import Profiler

class YoloFastAndSlow(nn.Module):
    def __init__(self, 
                 names,
                 weights_small="yolo11n.pt",
                 weights_large="yolo11x.pt",
                 use_large_head = False,
                 device="cuda"):
        super().__init__()
        self.device = device

        small_model = YOLO(weights_small)
        large_model = YOLO(weights_large)

        self.backbone_small = small_model.model.to(device).eval()
        self.backbone_large = large_model.model.to(device).eval()
        
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

        self.profiler = Profiler(10)
        # self.profiler = None

        for p in self.parameters():
            p.requires_grad = True


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
    
    def profile(self, key: str):
        if(self.profiler is not None):
            self.profiler.process(key)

    def profiler_iteration_start(self):
        if(self.profiler is not None):
            self.profiler.iteration_start()

    def profiler_iteration_end(self):
        if(self.profiler is not None):
            self.profiler.iteration_end()

    def forward_frame(self, frames: torch.Tensor):
        self.profiler_iteration_start()

        x_t = frames
        backbone, adapter, is_large = self.choose_cur_backbone_adapter()
        self.profile("Backbone choose")

        with torch.no_grad():
            feats = self._extract_feats(backbone, x_t)

        self.profile("Extracting features")

        adapted = adapter(feats)

        self.profile("Adapter")

        if self.state is None:
            self.state = self.temporal.init_states(adapted)

        self.profile("State init")

        out_feats, state = self.temporal.step(adapted, self.state)
        if(is_large):
            self.state = state

        self.profile("Temporal")
        
        y = self.detect.forward(out_feats)

        self.profile("Head")
        
        self.current_t += 1
        res = self.postprocess(y[0], x_t)

        self.profile("Postprocess")

        self.profiler_iteration_end()

        return res
    
    def forward_seq(self, frames: torch.Tensor):
        self.profiler_iteration_start()

        B, T, C, H, W = frames.shape
        outputs = []
        states = None

        for t in range(T):
            x_t = frames[:, t]
            backbone, adapter, is_large = self.choose_backbone_adapter(t)

            self.profile("Backbone choose")

            with torch.no_grad():
                feats = self._extract_feats(backbone, x_t)
            
            self.profile("Extracting features")

            adapted = adapter(feats)

            self.profile("Adapter")

            if states is None:
                states = self.temporal.init_states(adapted)

            self.profile("State init")

            out_feats, states_after = self.temporal.step(adapted, states)
            if(is_large):
                states = states_after

            self.profile("Temporal")
                
            y = self.detect.forward(out_feats)

            self.profile("Head")

            if(self.detect.training):
                outputs.append(y)
            else:
                outputs.append(self.postprocess(y[0], x_t))

            self.profile("Postprocess")

        self.profiler_iteration_end()

        return outputs

    def freeze(self, key: str, freeze_state: bool):
        if(key == "backbone"):
            for p in self.backbone_small.parameters():
                p.requires_grad = not freeze_state
            for p in self.backbone_large.parameters():
                p.requires_grad = not freeze_state
        elif(key == "temporal"):
            for p in self.adapter_small.parameters():
                p.requires_grad = not freeze_state
            for p in self.adapter_large.parameters():
                p.requires_grad = not freeze_state
            for p in self.temporal.parameters():
                p.requires_grad = not freeze_state
        elif(key == "head"):
            for p in self.detect.parameters():
                p.requires_grad = not freeze_state
