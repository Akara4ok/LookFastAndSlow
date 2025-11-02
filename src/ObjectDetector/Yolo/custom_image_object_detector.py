import math
import os
import time
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import ultralytics
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.modules.head import Detect

from ObjectDetector.Yolo.general_image_object_detector import GeneralImageObjectDetector


class CustomImageObjectDetector(GeneralImageObjectDetector):
    def __init__(self, config: Dict, labels: List[str], map_classes = None, device: torch.device | str | None = None):
        super().__init__(config, labels, map_classes, device)

    def collate(self, batch):
        imgs, boxes, labels = [], [], []
        for img, tgt in batch:
            imgs.append(img)
            boxes.append(torch.as_tensor(tgt["boxes"], dtype=torch.float32))
            labels.append(torch.as_tensor(tgt["labels"], dtype=torch.float32))

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
        head = self.model.model.model[-1]
        nc = int(getattr(head, "nc", len(self.labels)))
        for i, (xywh, labels) in enumerate(zip(batch["boxes"], batch["labels"])):
            if(xywh.shape[0] != labels.shape[0]):
                raise ValueError(f"[prep_batch] sample {i}: boxes({xywh.shape[0]}) != labels({cls.shape[0]})")

            if labels.numel() > 0:
                minc, maxc = int(labels.min().item()), int(labels.max().item())
                if minc < 0 or maxc >= nc:
                    raise ValueError(
                        f"[prep_batch] sample {i}: class id out of range [0,{nc-1}]. "
                        f"min={minc}, max={maxc}. Перевір свій датасет (індекси мають бути 0-based)."
                    )


            xywh = xywh.to(self.device)
            labels = labels.to(self.device)
            bboxes_list.append(xywh)
            cls_list.append(labels.float().unsqueeze(1))
            if xywh.shape[0] > 0:
                batch_idx.append(torch.full((xywh.shape[0],), i, device=self.device, dtype=torch.int64))


        bboxes = torch.cat(bboxes_list, dim=0) if len(bboxes_list) else imgs.new_zeros((0, 4))
        cls = torch.cat(cls_list, dim=0) if len(cls_list) else imgs.new_zeros((0, 1))
        batch_idx = torch.cat(batch_idx, dim=0) if len(batch_idx) else torch.zeros((0,), device=self.device, dtype=torch.int64)

        M = bboxes.shape[0]
        if not (cls.shape[0] == M and batch_idx.shape[0] == M):
            raise RuntimeError(f"[prep_batch] inconsistent: M={M}, cls={cls.shape[0]}, bi={batch_idx.shape[0]}")
        
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

        writer = SummaryWriter(self.config["train"]["tensorboard_path"])

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
                total_norm = torch.nn.utils.clip_grad_norm_(model_core.parameters(), max_norm=1000.0)
                assert not torch.isnan(total_norm)
                optimizer.step()
                cur_loss += float(loss.mean().detach().item())
                
                # print(f"Step {i}, loss {loss.mean().detach().item()}")

            epoch_loss = cur_loss / len(train_loader)
            took = time.time() - t0

            val_loss = self._eval_loss(model_core, val_loader)

            writer.add_scalar("loss/train", epoch_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            print(f"[Epoch {epoch:03d}/{epochs}] "
                  f"train_loss={epoch_loss:.4f} "
                  f"val_loss={val_loss:.4f}"
                  f" time={took:.1f}s")
            
            if(best_val_loss > val_loss):
                best_val_loss = val_loss
                self.save_checkpoint(model_core, self.config['model']['path'])
            
                print(f"  ↳ new best, saved to {self.config['model']['path']}");

        model_core.eval()

    @torch.no_grad()
    def _eval_loss(self, model_core: nn.Module, val_loader: DataLoader) -> float:
        model_core.eval()
        totals, n = 0.0, 0
        for batch in val_loader:
            ultra_batch = self._prep_batch_for_ultra(batch)
            out = model_core.loss(ultra_batch)
            loss = out[0] if isinstance(out, (tuple, list)) else out
            totals += float(loss.mean().detach().item())
            n += 1
        return totals / n

    def save_checkpoint(self, net, path):
        torch.save({"model": net, "state_dict": net.state_dict()}, path)
