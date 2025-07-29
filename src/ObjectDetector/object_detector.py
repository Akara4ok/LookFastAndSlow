import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from ObjectDetector.Models.ssd_lite import SSDLite
from ObjectDetector.Anchors.anchors import Anchors, AnchorSpec
from ObjectDetector.loss import SSDLoss
from ObjectDetector.postprocessing import PostProcessor


class ObjectDetector:
    def __init__(self, labels: List[str], config: Dict, specs: List[AnchorSpec], device: torch.device | str | None = None):
        self.cfg = config
        self.labels = labels
        self.device = torch.device(device or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        a_cfg = self.cfg["anchors"]
        self.anchors = Anchors(specs, self.cfg["model"]["img_size"], a_cfg["variances"], device=self.device)

        self.model = SSDLite(config['model']['img_size'], len(labels), self.anchors.aspects).to(self.device)

        self.check_dims()

        self.post = PostProcessor(self.anchors,
                                  conf_thresh=a_cfg["confidence"],
                                  iou_thresh=a_cfg["iou_threshold"],
                                  top_k=a_cfg["top_k_classes"])

        self.criterion = SSDLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg["lr"]["initial_lr"])
        self.writer = SummaryWriter(self.cfg["train"]["tensorboard_path"])

    def check_dims(self):
        n_anchors = self.anchors.corner_anchors.size(0)
        test_tensor = torch.zeros(2, 3, self.cfg["model"]["img_size"], self.cfg["model"]["img_size"], device=self.device)
        cls_out = self.model(test_tensor)[1]
        if cls_out.size(1) != n_anchors:
            logging.warning(f"Model produces {cls_out.size(1)} anchors "
                            f"but generator made {n_anchors}")


    def _encode_sample(self,
                       img: torch.Tensor,          # (3,H,W)
                       target: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes  = target["boxes"].to(self.device)
        labels = target["labels"].to(self.device)

        loc_t, labels = self.anchors.match(boxes, labels)

        return img.to(self.device), labels, loc_t

    def _collate(self, batch: torch.Tensor):
        imgs, cls_t, loc_t = [], [], []
        for img, tgt in batch:
            i, c, l = self._encode_sample(img, tgt)
            imgs.append(i)
            cls_t.append(c)
            loc_t.append(l)

        return (torch.stack(imgs), torch.stack(cls_t), torch.stack(loc_t))

    def _split_datasets(self, full_ds: Dataset, test_ratio: float):
        test_len = int(len(full_ds) * test_ratio)
        train_len = len(full_ds) - test_len
        return random_split(full_ds, [train_len, test_len])

    def train(self, ds):
        ds_train, ds_val = self._split_datasets(ds, self.cfg["data"]["test_percent"])

        dl_train = DataLoader(ds_train,
                              batch_size=self.cfg["train"]["batch_size"],
                              shuffle=True,
                              collate_fn=self._collate,
                              pin_memory=True, drop_last=True)

        dl_val   = DataLoader(ds_val,
                              batch_size=self.cfg["train"]["batch_size"],
                              shuffle=False,
                              num_workers=0,
                              collate_fn=self._collate,
                              pin_memory=True, drop_last=True)

        best_val = float("inf")
        ckpt_dir = Path(self.cfg["model"]["path"]).expanduser().parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg["train"]["epochs"]):
            t0 = time.time()
            train_loss, train_loc_loss, train_cls_loss = self._run_epoch(dl_train, True)
            val_loss, val_loc_loss, val_cls_loss = self._run_epoch(dl_val, False)
            dt = time.time() - t0

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/train_loc", train_loc_loss, epoch)
            self.writer.add_scalar("loss/train_cls", train_cls_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss,   epoch)
            self.writer.add_scalar("loss/val_loc", val_loc_loss, epoch)
            self.writer.add_scalar("loss/val_cls", val_cls_loss, epoch)

            logging.info(f"Epoch {epoch:03d} time {dt:.1f}s")
            logging.info(f"train {train_loss:.4f} loc {train_loc_loss:.4f} cls {train_cls_loss:.4f}")
            logging.info(f"val {val_loss:.4f} loc {val_loc_loss:.4f} cls {val_cls_loss:.4f}")
            logging.info(f"===================")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), self.cfg["model"]["path"])
                logging.info(f"  ↳ new best, saved to {self.cfg['model']['path']}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, n_batches = 0.0, 0
        total_loc, total_cls = 0.0, 0.0

        with torch.set_grad_enabled(train):
            for imgs, cls_gt, loc_gt in loader:
                imgs  = imgs.to(self.device)
                cls_gt = cls_gt.to(self.device)
                loc_gt = loc_gt.to(self.device)

                loc_p, cls_p = self.model(imgs)

                loc_loss, cls_loss = self.criterion(loc_p, cls_p, loc_gt, cls_gt)
                loss = loc_loss + cls_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                total_loc += loc_loss.item()
                total_cls += cls_loss.item()
                n_batches  += 1

        return (total_loss / n_batches, total_loc / n_batches, total_cls / n_batches) 

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)

        self.model.eval()
        with torch.no_grad():
            loc_p, cls_p = self.model(img)

        result = self.post.ssd_postprocess(cls_p.cpu(), loc_p.cpu())
        return result