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

from ObjectDetector.Yolo.general_video_object_detector import GeneralVideoObjectDetector
from ObjectDetector.Yolo.Models.yolo_fast_and_slow import YoloFastAndSlow


class CustomVideoObjectDetector(GeneralVideoObjectDetector):
    def __init__(self, config: Dict, labels: List[str], inference = False, device: torch.device | str | None = None):
        super().__init__(config, labels, None, device)
        self.model : YoloFastAndSlow = None
        self.inference = inference

    def load_weights(self, weights_path: str, small: str = None, large: str = None, use_large_head: bool = True):
        self.model = YoloFastAndSlow(self.labels, small, large, use_large_head)
        logging.info(f"Creating model, large from {large}, small from {small} and use large head: {use_large_head}")
        if(weights_path is not None):
            self.model.load_state_dict(torch.load(weights_path))
            self.model.seq = not self.inference
            if(self.inference):
                self.model.eval()

            logging.info(f"Loading model from {weights_path}, inference: {self.inference}")

    def collate(self, batch):
        imgs = [img_seq for img_seq, _ in batch] # list of (T,C,H,W)
        imgs = torch.stack(imgs, dim=0) # (B,T,C,H,W)

        boxes_batch = []
        labels_batch = []
        for _, tgts in batch:
            seq_boxes, seq_labels = [], []
            for frame_tgt in tgts:
                seq_boxes.append(frame_tgt["boxes"])
                seq_labels.append(frame_tgt["labels"])
            boxes_batch.append(seq_boxes)
            labels_batch.append(seq_labels)

        return {
            "images": imgs,
            "boxes": boxes_batch,
            "labels": labels_batch,
        }
    

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
            return DataLoader(
                dataset,
                batch_size=self.config["train"]["batch_size"],
                shuffle=shuffle,
                collate_fn=self.collate,
                drop_last=False
            )
    
    def _prep_batch_for_ultra_video(self, batch: Dict) -> Dict:
        imgs = batch["images"].to(self.device)  # (B,T,C,H,W)
        B, T, C, H, W = imgs.shape

        nc = len(self.labels)

        gt_per_frame = []

        # Iterate over frames
        for t in range(T):
            bboxes_list = []
            cls_list = []
            batch_idx = []

            for b in range(B):
                xywh = batch["boxes"][b][t]
                labels = batch["labels"][b][t]

                if xywh.shape[0] != labels.shape[0]:
                    raise ValueError(f"[prep_batch] seq {b}, frame {t}: boxes({xywh.shape[0]}) != labels({labels.shape[0]})")

                if labels.numel() > 0:
                    minc, maxc = int(labels.min().item()), int(labels.max().item())
                    if minc < 0 or maxc >= nc:
                        raise ValueError(
                            f"[prep_batch] seq {b}, frame {t}: class id out of range [0,{nc-1}]"
                        )

                xywh = xywh.to(self.device)
                labels = labels.to(self.device)
                bboxes_list.append(xywh)
                cls_list.append(labels.float().unsqueeze(1))

                if xywh.shape[0] > 0:
                    batch_idx.append(torch.full((xywh.shape[0],), b, device=self.device, dtype=torch.int64))

            bboxes = torch.cat(bboxes_list, dim=0) if bboxes_list else imgs.new_zeros((0, 4))
            cls = torch.cat(cls_list, dim=0) if cls_list else imgs.new_zeros((0, 1))
            batch_idx = torch.cat(batch_idx, dim=0) if batch_idx else torch.zeros((0,), device=self.device, dtype=torch.int64)

            M = bboxes.shape[0]
            if not (cls.shape[0] == M and batch_idx.shape[0] == M):
                raise RuntimeError(f"[prep_batch] inconsistent frame {t}: M={M}, cls={cls.shape[0]}, bi={batch_idx.shape[0]}")

            frame_gt = {
                "imgs": imgs[:, t].to(self.device),  # (B,C,H,W)
                "bboxes": bboxes,
                "cls": cls,
                "batch_idx": batch_idx,
            }
            gt_per_frame.append(frame_gt)

        return {
            "imgs": imgs,  # (B,T,C,H,W)
            "gt": gt_per_frame
        }

    def _split_datasets(self, full_ds: Dataset, test_ratio: float):
        test_len = int(len(full_ds) * test_ratio)
        train_len = len(full_ds) - test_len
        
        return random_split(full_ds, [train_len, test_len])

    def train(self, train_dataset: Dataset, val_dataset: Dataset = None, freeze_dict: dict = None):
        logging.info("Training started")
        if(val_dataset is None):
            train_dataset, val_dataset = self._split_datasets(train_dataset, self.config["data"]["test_percent"])
        
        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader = self._make_loader(val_dataset, shuffle=False)

        self._train_loop_pure_torch(train_loader, val_loader, freeze_dict)

    def freeze_handle(self, epoch: int, freeze_dict: dict, optimizer: torch.optim.Optimizer):
        '''
        dict: {
            "key": (freeze_start, freeze_end, lr)
            key: backbone, temporal, head 
        }
        '''
        if(freeze_dict is None):
            return optimizer
        
        updated = False
        cur_lr = optimizer.param_groups[0]["lr"]
        for key, (start, end, lr) in freeze_dict.items():
            if epoch == start:
                self.model.freeze(key, True)
                updated = True
                logging.info(f"{key} was freezed on {epoch} epoch")
            if epoch == end:
                self.model.freeze(key, False)
                updated = True
                if(lr is not None):
                    cur_lr = lr
                logging.info(f"{key} was unfreezed on {epoch} epoch with {cur_lr:.5f} lr"))

        if(not updated):
            return optimizer
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cur_lr)
        return optimizer


    def _train_loop_pure_torch(self, train_loader, val_loader, freeze_dict):
        from types import SimpleNamespace

        self.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        loss_fn = v8DetectionLoss(self.model)

        epochs = self.config["train"]["epochs"]
        lr = self.config["lr"]["initial_lr"]

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")

        writer = SummaryWriter(self.config["train"]["tensorboard_path"])

        for epoch in range(1, epochs + 1):
            optimizer = self.freeze_handle(epoch, freeze_dict, optimizer)
            self.model.train()
            cur_loss = 0.0
            t0 = time.time()

            for i, batch in enumerate(train_loader):
                ultra_batch = self._prep_batch_for_ultra_video(batch)

                imgs_seq = ultra_batch["imgs"].to(self.device)          # (B, T, C, H, W)
                gt_frames = ultra_batch["gt"]                           # list of T ground-truth dicts

                # forward
                preds = self.model(imgs_seq)                            # list[T] of predictions
                assert len(preds) == len(gt_frames), f"Mismatch: preds={len(preds)}, gt={len(gt_frames)}"

                total_loss = 0.0
                optimizer.zero_grad()

                for t, (pred, gt_t) in enumerate(zip(preds, gt_frames)):
                    out = loss_fn(pred, gt_t)
                    loss = out[0] if isinstance(out, (tuple, list)) and len(out) >= 1 else out
                    total_loss += loss.mean()

                total_loss.backward()
                # total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1000.0)
                # assert not torch.isnan(total_norm)
                optimizer.step()

                batch_loss = float(total_loss.detach().item() / len(gt_frames)) / imgs_seq.shape[0]
                cur_loss += batch_loss

                if(i % 100 == 0):
                    print(f"Step {i}, loss {batch_loss}")
                
                break

            epoch_loss = cur_loss / len(train_loader)
            val_loss = self._eval_loss(self.model, val_loader)

            took = time.time() - t0

            writer.add_scalar("loss/train", epoch_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            print(f"[Epoch {epoch:03d}/{epochs}] "
                  f"train_loss={epoch_loss:.4f} "
                  f"val_loss={val_loss:.4f}"
                  f" time={took:.1f}s")
            
            if(best_val_loss > val_loss):
                best_val_loss = val_loss
                self.save_checkpoint(self.model, self.config['model']['path'])
            
                print(f"  ↳ new best, saved to {self.config['model']['path']}")
            
            break

        self.model.eval()

    @torch.no_grad()
    def _eval_loss(self, model_core: nn.Module, val_loader: DataLoader) -> float:
        totals, n = 0.0, 0

        for batch in val_loader:
            ultra_batch = self._prep_batch_for_ultra_video(batch)
            imgs_seq = ultra_batch["imgs"].to(self.device)   # (B,T,C,H,W)
            gt_frames = ultra_batch["gt"]                    # list[T] of dicts

            # Forward pass (returns list[T] of predictions)
            preds = model_core(imgs_seq)
            assert len(preds) == len(gt_frames), \
                f"Mismatch: preds={len(preds)}, gt={len(gt_frames)}"

            # Compute mean loss over all frames in sequence
            total_loss = 0.0
            for pred, gt_t in zip(preds, gt_frames):
                out = model_core.loss(pred, gt_t)
                loss = out[0] if isinstance(out, (tuple, list)) else out
                total_loss += loss.mean()

            total_loss = total_loss / len(gt_frames) / imgs_seq.shape[0]
            totals += float(total_loss.detach().item())
            n += 1

        return totals / max(n, 1)

    def save_checkpoint(self, net, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(net.state_dict(), path)
