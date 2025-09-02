import time
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from ObjectDetector.Models.interleaved_classifier import InterleavedClassifier
from ObjectDetector.loss import SSDLoss

class Phase1Trainer:
    def __init__(self, num_classes: int, config: Dict, device: torch.device | str | None = None):
        self.cfg = config
        self.device = torch.device(device or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        
        model_cfg = config["model"]
        self.model = InterleavedClassifier(model_cfg["fast_width"], model_cfg["slow_width"], model_cfg["backbone_out_channels"], model_cfg["lstm_out_channels"], num_classes)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg["lr"]["initial_lr"], weight_decay=5e-4)
        self.writer = SummaryWriter(self.cfg["train"]["tensorboard_path"])

    def _collate(self, batch: torch.Tensor):
        imgs_seqs, labels = zip(*batch)
        x = torch.stack(imgs_seqs, dim=0)
        y = torch.as_tensor(labels, dtype=torch.long)
        return x, y

    def train(self, ds_train, ds_val):
        logging.info("Training started")
        dl_train = DataLoader(ds_train,
                              batch_size=self.cfg["train"]["batch_size"],
                              shuffle=True,
                              collate_fn=self._collate,
                              drop_last=True)

        dl_val   = DataLoader(ds_val,
                              batch_size=self.cfg["train"]["batch_size"],
                              shuffle=False,
                              num_workers=0,
                              collate_fn=self._collate,
                              drop_last=False)

        best_val = float("inf")
        ckpt_dir = Path(self.cfg["model"]["path"]).expanduser().parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg["train"]["epochs"]):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(dl_train, True)
            val_loss, val_acc = self._run_epoch(dl_val, False)
            dt = time.time() - t0

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/train_acc", train_acc, epoch)
            self.writer.add_scalar("loss/val", val_loss,   epoch)
            self.writer.add_scalar("loss/val_acc", val_acc, epoch)

            logging.info(f"Epoch: {epoch}, time: {dt}s")
            logging.info(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
            logging.info(f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")
            logging.info(f"===================")

            if val_loss > best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), self.cfg["model"]["path"])
                logging.info(f"  ↳ new best, saved to {self.cfg['model']['path']}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        T = loader.dataset.seq_len
        
        for imgs_seq, labels in loader:
            imgs_seq = imgs_seq.to(self.device)  # (B,T,C,H,W)
            labels = labels.to(self.device)      # (B,)

            if train:
                self.optimizer.zero_grad()

            logits_seq = self.model.forward(imgs_seq)  # (B,T,num_classes)
            loss = sum(F.cross_entropy(logits_seq[:, t], labels) for t in range(T)) / T

            if train:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            total_loss += loss.item() * imgs_seq.size(0)
            preds = logits_seq[:, -1].argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        return avg_loss, acc

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
