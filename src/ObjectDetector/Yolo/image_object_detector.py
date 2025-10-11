import math
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import ultralytics
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.modules.head import Detect

from ObjectDetector.map import MeanAveragePrecision
from Dataset.Yolo.YoloTestDataset import YoloTestDataset

class ImageObjectDetector:
    def __init__(self, labels: List[str], config: Dict, map_classes = None, device: torch.device | str | None = None):
        self.labels = labels
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = None
        self.map_classes = map_classes

    def load_weights(self, weights_path: str, base: str = None):
        if base is None:
            self.model = YOLO(weights_path)

            if self.labels is not None and len(self.labels) != len(self.model.names) and self.map_classes is None:
                self.model = self.set_num_classes_yolo11(self.model, self.labels)
            
            if self.labels is None:
                self.labels = self.model.names

            self.model.to(self.device)

            logging.info(f"Model loaded to {self.device} from {weights_path}")
        else:
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.model = YOLO(base)

            self.model = self.set_num_classes_yolo11(self.model, self.labels)

            missing, unexpected = self.model.model.load_state_dict(ckpt["state_dict"])
            if missing or unexpected:
                print("Missing keys:", missing)
                print("Unexpected keys:", unexpected)

            self.model.to(self.device).eval()
            logging.info(f"Model loaded to {self.device} from {weights_path} and adapted from {base}")


    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        res = self.model.predict(frame, verbose=False)

        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "classes": np.zeros((0,), np.int64)}

        boxes_xyxy = r0.boxes.xyxyn.detach().cpu().numpy().astype(np.float32)
        scores = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = r0.boxes.cls.detach().cpu().numpy().astype(np.int64)
        predicted = {"boxes": boxes_xyxy, "scores": scores, "classes": classes}
        return self.map_result_classes(predicted)
    
    def map_result_classes(self, predicted: dict) -> dict:
        if(self.map_classes is None):
            return predicted
        
        mapped_boxes = []
        mapped_scores = []
        mapped_classes = []
        for box, score, cls_id in zip(predicted["boxes"], predicted["scores"], predicted["classes"]):
            if cls_id in self.map_classes:
                mapped_boxes.append(np.expand_dims(box, axis=0))
                mapped_scores.append(np.expand_dims(score, axis=0))
                mapped_classes.append(self.map_classes[cls_id])

        if len(mapped_boxes) > 0:
            boxes_out = np.concatenate(mapped_boxes, axis=0)
            scores_out = np.concatenate(mapped_scores, axis=0)
            classes_out = np.stack(mapped_classes, axis=0)
        else:
            boxes_out = np.empty((0, 4), dtype=np.float32)
            scores_out = np.empty((0,), dtype=np.float32)
            classes_out = np.empty((0,), dtype=np.int64)

        return dict(boxes=boxes_out, scores=scores_out, classes=classes_out)
    
    def test(self, ds: Dataset, count: int) -> float:
        logging.info("Testing started")
        ds = YoloTestDataset(ds, self.config["model"]["img_size"])
        dl_test = DataLoader(ds,
                              batch_size=self.config["train"]["batch_size"],
                              shuffle=False,
                              num_workers=0,
                              collate_fn=self.test_collate,
                              drop_last=False)

        test_map = self._test_model(dl_test, count)
        return test_map
                
    def _test_model(self, dl_test: DataLoader, count: int) -> float:
        self.model.eval()

        metric = MeanAveragePrecision(num_classes=len(self.labels), device=self.device)
        metric.reset()

        current = 0

        with torch.set_grad_enabled(False):
            for batch in dl_test:
                imgs  = batch["images"].to(self.device)

                preds_batch = self.predict(imgs)
                tensor_dict = {k: torch.from_numpy(v) for k, v in preds_batch.items()}

                metric.update([tensor_dict], batch["raw"])
                
                current += self.config["train"]["batch_size"]
                if(current >= count):
                    break

        res = metric.compute()
        
        return  res["mAP"]

    def collate(self, batch):
        imgs, boxes, labels = [], [], []
        for img, tgt in batch:
            imgs.append(img)
            boxes.append(torch.as_tensor(tgt["boxes"], dtype=torch.float32))
            labels.append(torch.as_tensor(tgt["labels"], dtype=torch.float32))

        images = torch.stack(imgs, dim=0)  # [B,C,H,W]
        return {"images": images, "boxes": boxes, "labels": labels}
    
    def test_collate(self, batch):
        imgs, boxes, labels, raw = [], [], [], []
        for img, tgt in batch:
            imgs.append(img)
            boxes.append(torch.as_tensor(tgt["boxes"], dtype=torch.float32))
            labels.append(torch.as_tensor(tgt["labels"], dtype=torch.float32))
            raw.append(tgt)

        images = torch.stack(imgs, dim=0)  # [B,C,H,W]
        return {"images": images, "boxes": boxes, "labels": labels, "raw": raw}
    
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
                raise ValueError(f"[prep_batch] sample {i}: boxes({boxes.shape[0]}) != labels({cls.shape[0]})")

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
                print(f"grad_norm={float(total_norm):.2f}")
                optimizer.step()
                cur_loss += float(loss.mean().detach().item())
                
                print(f"Step {i}, loss {loss.mean().detach().item()}")

            epoch_loss = cur_loss / len(train_loader)
            took = time.time() - t0

            val_loss = self._eval_loss(model_core, val_loader)

            writer.add_scalar("loss/train", epoch_loss, epoch)
            writer.add_scalar("loss/мфд", val_loss, epoch)
            print(f"[Epoch {epoch:03d}/{epochs}] "
                  f"train_loss={epoch_loss:.4f} "
                  f"val_loss={val_loss:.4f}"
                  f" time={took:.1f}s")
            
            if(best_val_loss > val_loss):
                best_val_loss = val_loss
                self.save_checkpoint(model_core, self.labels, self.config['model']['path'])
            
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


    def set_num_classes_yolo11(self, model_module: nn.Module, class_names):
        new_nc = len(class_names)

        head = None
        for m in model_module.modules():
            if isinstance(m, Detect):
                head = m

        head.nc = new_nc

        def _swap_last_conv_to_nc(seq: nn.Sequential, nc: int):
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

    def save_checkpoint(self, net, class_names, path):
        head = next(m for m in net.modules() if m.__class__.__name__.lower().endswith("detect"))
        ckpt = {
            "state_dict": net.state_dict(),
            "names": list(class_names),
            "nc": len(class_names),
            "reg_max": getattr(head, "reg_max", None),
            "stride": getattr(head, "stride", None),
            "imgsz": 640,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ultralytics_version": getattr(ultralytics, "__version__", None),
            "torch_version": torch.__version__,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(ckpt, path)
