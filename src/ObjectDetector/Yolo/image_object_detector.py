import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.modules.head import Detect
import math

class ImageObjectDetector:
    def __init__(self, labels: List[str], config: Dict, device: torch.device | str | None = None):
        self.labels = labels
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = None

    def load_weights(self, weights_path: str):
        self.model = YOLO(weights_path)

        if self.labels is not None and len(self.labels) != len(self.model.names):
            self.model = self.set_num_classes_yolo11(self.model, self.labels)

        self.model.to(self.device)

        logging.info(f"Model loaded to {self.device} from {weights_path}")


    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        res = self.model.predict(frame, verbose=False, device=self.device)

        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "classes": np.zeros((0,), np.int64)}

        boxes_xyxy = r0.boxes.xyxyn.detach().cpu().numpy().astype(np.float32)
        scores = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = r0.boxes.cls.detach().cpu().numpy().astype(np.int64)
        return {"boxes": boxes_xyxy, "scores": scores, "classes": classes}

    def collate(self, batch):
        imgs, boxes, labels = [], [], []
        for img, tgt in batch:
            imgs.append(img)
            boxes.append(torch.as_tensor(tgt["boxes"], dtype=torch.float32))   # (Ni,4) XYXY у пікселях
            labels.append(torch.as_tensor(tgt["labels"], dtype=torch.float32)) # (Ni,)  індекси класів

        images = torch.stack(imgs, dim=0)  # [B,C,H,W]
        return {"images": images, "boxes": boxes, "labels": labels}
    
    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
            return DataLoader(
                dataset,
                batch_size=self.config["train"]["batch_size"],
                shuffle=shuffle,
                collate_fn=self.collate,
                drop_last=False
            )

    def _prep_batch_for_ultra(self, batch: Dict) -> Dict:
        imgs = batch["images"].to(self.device)  # (B,C,H,W), float in [0,1]
        B, _, H, W = imgs.shape
        bboxes_list = []
        cls_list = []
        batch_idx = []
        for i, (xywh, labels) in enumerate(zip(batch["boxes"], batch["labels"])):
            xywh = xywh.to(self.device)
            labels = labels.to(self.device)
            bboxes_list.append(xywh)
            cls_list.append(labels.float().unsqueeze(1))
            if xywh.shape[0] > 0:
                batch_idx.append(torch.full((xywh.shape[0],), i, device=self.device, dtype=torch.int64))


        bboxes = torch.cat(bboxes_list, dim=0) if len(bboxes_list) else imgs.new_zeros((0, 4))
        cls = torch.cat(cls_list, dim=0) if len(cls_list) else imgs.new_zeros((0, 1))
        batch_idx = torch.cat(batch_idx, dim=0) if len(batch_idx) else torch.zeros((0,), device=self.device, dtype=torch.int64)

        return {
            "img": imgs,              # (B,C,H,W) float
            "bboxes": bboxes,         # (M,4)
            "cls": cls,               # (M,1)
            "batch_idx": batch_idx,   # (M,)
        }
    
    def _split_datasets(self, full_ds: Dataset, test_ratio: float):
        test_len = int(len(full_ds) * test_ratio)
        train_len = len(full_ds) - test_len
        
        return random_split(full_ds, [train_len, test_len])
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset = None):
        logging.info("Training started")
        if(val_dataset is None):
            train_dataset, val_dataset = self._split_datasets(train_dataset, self.config["data"]["test_percent"])
        
        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader = self._make_loader(val_dataset, shuffle=False)

        self._train_loop_pure_torch(train_loader, val_loader)

    def _train_loop_pure_torch(self, train_loader, val_loader):
        from types import SimpleNamespace
        model_core = self.model.model
        model_core.to(self.device).train()
        model_core.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        loss_fn = v8DetectionLoss(model_core)

        epochs = self.config["train"]["epochs"]
        lr = self.config["lr"]["initial_lr"]

        optimizer = torch.optim.AdamW(model_core.parameters(), lr=lr)

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model_core.train()
            cur_loss = 0.0
            t0 = time.time()

            for i, batch in enumerate(train_loader):
                ultra_batch = self._prep_batch_for_ultra(batch)
                # print("img", ultra_batch["img"].shape,
                #     "b", ultra_batch["bboxes"].shape,
                #     "c", ultra_batch["cls"].shape,
                #     "bi", ultra_batch["batch_idx"].shape,
                #     "args_type", type(model_core.args),
                #     "has_box", hasattr(model_core.args, "box") or ("box" in model_core.args))

                preds = model_core(ultra_batch["img"].requires_grad_(True))
                out = loss_fn(preds, ultra_batch)
                loss = out[0] if isinstance(out, (tuple, list)) and len(out) >= 1 else out
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
                cur_loss += float(loss.mean().detach().item())
                print(f"Step {i}, loss {loss.mean().detach().item()}")

            epoch_loss = cur_loss / len(train_loader)
            took = time.time() - t0

            val_loss = self._eval_loss(model_core, val_loader)
            best_val_loss = min(best_val_loss, val_loss)

            print(f"[Epoch {epoch:03d}/{epochs}] "
                  f"train_loss={epoch_loss:.4f} "
                  f"{' val_loss=' + format(val_loss, '.4f') if val_loss is not None else ''} "
                  f" time={took:.1f}s")

        model_core.eval()

    @torch.no_grad()
    def _eval_loss(self, model_core: nn.Module, val_loader: DataLoader) -> float:
        model_core.eval()
        totals, n = 0.0, 0
        for batch in val_loader:
            ultra_batch = self._prep_batch_for_ultra(batch)
            out = model_core.loss(ultra_batch)
            loss = out[0] if isinstance(out, (tuple, list)) else out
            totals += float(loss.item())
            n += 1
        return totals / n


    def set_num_classes_yolo11(self, model_module: nn.Module, class_names):
        new_nc = len(class_names)

        head = None
        for m in model_module.modules():
            if isinstance(m, Detect):
                head = m
        if head is None:
            raise RuntimeError("Detect head not found. Is this a YOLO11 Detect model?")

        head.nc = new_nc

        def _swap_last_conv_to_nc(seq: nn.Sequential, nc: int):
            assert isinstance(seq[-1], nn.Conv2d), "Unexpected cls head structure"
            last = seq[-1]
            new = nn.Conv2d(last.in_channels, nc, kernel_size=1, bias=True)
            nn.init.zeros_(new.weight)
            nn.init.constant_(new.bias, -math.log((1 - 0.01) / 0.01))
            seq[-1] = new

        for i, seq in enumerate(head.cv3):
            _swap_last_conv_to_nc(seq, new_nc)

        if hasattr(head, "reg_max"):
            head.no = head.nc + 4 * head.reg_max

        if hasattr(head, "bias_init"):
            head.bias_init()

        (getattr(model_module, "model", model_module)).names = list(class_names)

        return model_module
