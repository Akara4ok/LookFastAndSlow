import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ObjectDetector.Shared.Models.conv_lstm import MultiScaleConvLSTM, Adapter

from ObjectDetector.profiler import Profiler

from torchvision.ops import nms

class Box:
    def __init__(self, xyxyn, conf, cls):
        self.xyxyn = xyxyn
        self.conf = conf
        self.cls = cls
        
    def __len__(self):
        return self.xyxyn.shape[0]
    
class Results:
    def __init__(self, xyxyn, conf, cls):
        self.boxes = Box(xyxyn, conf, cls)


class YoloWrapper(nn.Module):
    def __init__(self, 
                 names,
                 model="yolo11n.pt",
                 device="cuda"):
        super().__init__()
        self.device = device

        model = YOLO(model)

        self.backbone = model.model.to(device).eval()
        
        dummy = torch.zeros(1, 3, 320, 320).to(device)
        with torch.no_grad():
            feats_s = self._extract_feats(self.backbone, dummy)

        in_chs = [f.shape[1] for f in feats_s]

        ref_head = model.model.model[-1]
        
        self.detect = Detect(nc=len(names), ch=in_chs).to(device)
        # with torch.no_grad():
        #     if hasattr(ref_head, "stride"):
        #         self.detect.stride = ref_head.stride.clone()
        #     if hasattr(ref_head, "anchors"):
        #         self.detect.anchors = ref_head.anchors.clone()
        #     state = ref_head.state_dict()
        #     self.detect.load_state_dict(state, strict=False)

        self.names = names
        self.args = None

        self.profiler = Profiler(10)
        # self.profiler = None

        for p in self.parameters():
            p.requires_grad = True

    def put_inference(self):
        self.eval()
        self.backbone.fuse()

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
    

    @torch.no_grad()
    def postprocess(self, preds: torch.Tensor, imgs: torch.Tensor,
                    conf_thres: float = 0.25, iou_thres: float = 0.45):
        # self.profiler_iteration_start()

        device = self.device

        B = preds.size(0)
        nc = preds.shape[1] - 4
        results = []

        # self.profile("preprocess")

        results = []
        for i in range(B):
            p = preds[i]

            boxes = p[0:4, :].T             # (8400, 4)
            class_logits = p[4:, :].T       # (8400, 20)

            conf, cls = class_logits.max(1) # по класах

            # self.profile("before stuff")
            idx = torch.where(conf > conf_thres)[0]
            # self.profile("some stuff")

            boxes = boxes.index_select(0, idx)
            conf  = conf.index_select(0, idx)
            cls   = cls.index_select(0, idx)

            if boxes.numel() == 0:
                results.append(Results(torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0,), device=device, dtype=torch.long)))
                continue


            xy = boxes.clone()
            xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  
            xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

            # self.profile("cycle preprocess")

            keep = nms(xy, conf, iou_thres)

            # self.profile("nms")

            xy = xy[keep]
            conf = conf[keep]
            cls = cls[keep]

            h, w = imgs[i].shape[1:]
            xyxyn = xy.clone()
            xyxyn[:, [0, 2]] /= w
            xyxyn[:, [1, 3]] /= h

            results.append(Results(xyxyn, conf, cls))
            # self.profile("creating results")
        # self.profiler_iteration_end()
        return results

    def forward(self, frames: torch.Tensor):
        self.profiler_iteration_start()

        x_t = frames
        with torch.no_grad():
            feats = self._extract_feats(self.backbone, x_t)

        self.profile("Extracting features")

        y = self.detect.forward(feats)

        self.profile("Head")
        
        res = self.postprocess(y[0], x_t)

        self.profile("Postprocess")

        self.profiler_iteration_end()

        return res
        
    def predict(self, frames: torch.Tensor):
        return self.forward(frames)
    
    def profile(self, key: str):
        if(self.profiler is not None):
            torch.cuda.synchronize()
            self.profiler.process(key)

    def profiler_iteration_start(self):
        if(self.profiler is not None):
            self.profiler.iteration_start()

    def profiler_iteration_end(self):
        if(self.profiler is not None):
            self.profiler.iteration_end()

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
