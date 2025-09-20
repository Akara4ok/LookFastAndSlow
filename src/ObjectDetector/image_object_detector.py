import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from Dataset.train_dataset import TrainDataset
from Dataset.test_dataset import TestDataset
from ObjectDetector.Models.ssd_lite import SSDLite
from ObjectDetector.Anchors.anchors import Anchors, AnchorSpec
from ObjectDetector.loss import SSDLoss
from ObjectDetector.postprocessing import PostProcessor
from ObjectDetector.map import MeanAveragePrecision
from torch.optim.lr_scheduler import CosineAnnealingLR

class ImageObjectDetector:
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
                                  iou_thresh=a_cfg["post_iou_threshold"],
                                  top_k=a_cfg["top_k_classes"])

        self.criterion = SSDLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg["lr"]["initial_lr"], weight_decay=5e-4)
        self.writer = SummaryWriter(self.cfg["train"]["tensorboard_path"])
        self.metric = MeanAveragePrecision(num_classes=len(self.labels), device=self.device)
        
        self.scheduler: CosineAnnealingLR = None
        min_lr = self.cfg["lr"]["min_lr"]
        if(min_lr is not None):
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg["train"]["epochs"], # number of epochs
                eta_min=1e-5                       # min LR
            )


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

        loc_t, labels = self.anchors.match(boxes, labels, self.cfg["anchors"]["iou_threshold"])

        return img.to(self.device), labels.to(self.device), loc_t.to(self.device)

    def _collate(self, batch: torch.Tensor):
        imgs, cls_t, loc_t, raw = [], [], [], []
        for img, tgt in batch:
            i, c, l = self._encode_sample(img, tgt)
            imgs.append(i)
            cls_t.append(c)
            loc_t.append(l)
            raw.append(tgt)

        return (torch.stack(imgs).to(self.device), torch.stack(cls_t).to(self.device), torch.stack(loc_t).to(self.device), raw)

    def _split_datasets(self, full_ds: Dataset, test_ratio: float):
        test_len = int(len(full_ds) * test_ratio)
        train_len = len(full_ds) - test_len
        img_size = self.cfg["model"]["img_size"]
        
        train_ds, val_ds = random_split(full_ds, [train_len, test_len])
        if(self.cfg["train"]["augmentation"]):
            return TrainDataset(train_ds, img_size), TestDataset(val_ds, img_size)
        return TestDataset(train_ds, img_size), TestDataset(val_ds, img_size)

    def train(self, ds):
        logging.info("Training started")
        ds_train, ds_val = self._split_datasets(ds, self.cfg["data"]["test_percent"])

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

        best_map = -1.0

        for epoch in range(self.cfg["train"]["epochs"]):
            compute_metric = (epoch % 5 == 0)
            
            t0 = time.time()
            train_loss, train_loc_loss, train_cls_loss, train_map = self._run_epoch(dl_train, True, compute_metric)
            val_loss, val_loc_loss, val_cls_loss, val_map = self._run_epoch(dl_val, False, True)
            dt = time.time() - t0

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/train_loc", train_loc_loss, epoch)
            self.writer.add_scalar("loss/train_cls", train_cls_loss, epoch)
            self.writer.add_scalar("loss/train_map", train_map, epoch)
            self.writer.add_scalar("loss/val", val_loss,   epoch)
            self.writer.add_scalar("loss/val_loc", val_loc_loss, epoch)
            self.writer.add_scalar("loss/val_cls", val_cls_loss, epoch)
            self.writer.add_scalar("loss/val_map", val_map, epoch)
            
            if(self.scheduler is not None):
                self.scheduler.step()
                logging.info(f"Epoch {epoch:03d} time {dt:.1f}s lr {self.scheduler.get_last_lr()[0]:.6f}")
            else:
                logging.info(f"Epoch {epoch:03d} time {dt:.1f}s")

            log_metric = f"train {train_loss:.4f} loc {train_loc_loss:.4f} cls {train_cls_loss:.4f}"
            if(compute_metric):
                log_metric += f" map {train_map:.4f}"
            logging.info(log_metric)
            logging.info(f"val {val_loss:.4f} loc {val_loc_loss:.4f} cls {val_cls_loss:.4f} map {val_map:.4f}")
            logging.info(f"===================")

            if val_map > best_map:
                best_map = val_map
                torch.save(self.model.state_dict(), self.cfg["model"]["path"])
                logging.info(f"  ↳ new best, saved to {self.cfg['model']['path']}")

    def _run_epoch(self, loader: DataLoader, train: bool, compute_metric: bool) -> tuple:
        if train:
            self.model.train()
        else:
            self.model.eval()

        self.metric.reset()
        total_loss, n_batches = 0.0, 0
        total_loc, total_cls = 0.0, 0.0

        with torch.set_grad_enabled(train):
            for imgs, cls_gt, loc_gt, raw in loader:
                imgs  = imgs.to(self.device)
                cls_gt = cls_gt.to(self.device)
                loc_gt = loc_gt.to(self.device)

                loc_p, cls_p = self.model(imgs)
                loc_loss, cls_loss = self.criterion(loc_p, cls_p, loc_gt, cls_gt)
                loss = loc_loss + cls_loss

                if(compute_metric):
                    preds_batch = []
                    for cls_i, loc_i in zip(cls_p, loc_p):
                        preds_batch.append(self.post.ssd_postprocess(cls_i, loc_i))
                    self.metric.update(preds_batch, raw)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                total_loc += loc_loss.item()
                total_cls += cls_loss.item()
                
                n_batches += 1

        res = {"mAP": 0}
        if(compute_metric):
            res = self.metric.compute()
        
        val_mAP = res["mAP"]
        
        return (total_loss / n_batches, total_loc / n_batches, total_cls / n_batches, val_mAP) 

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
    def test(self, ds: Dataset, count: int) -> float:
        logging.info("Testing started")
        ds = TestDataset(ds, self.cfg["model"]["img_size"])
        dl_test   = DataLoader(ds,
                              batch_size=self.cfg["train"]["batch_size"],
                              shuffle=False,
                              num_workers=0,
                              collate_fn=self._collate,
                              drop_last=False)

        test_map = self._test_model(dl_test, count)
        return test_map
                
    def _test_model(self, dl_test: DataLoader, count: int) -> float:
        self.model.eval()

        self.metric.reset()

        current = 0

        with torch.set_grad_enabled(False):
            for imgs, cls_gt, loc_gt, raw in dl_test:
                imgs  = imgs.to(self.device)
                cls_gt = cls_gt.to(self.device)
                loc_gt = loc_gt.to(self.device)

                loc_p, cls_p = self.model(imgs)

                preds_batch = []
                for cls_i, loc_i in zip(cls_p, loc_p):
                    preds_batch.append(self.post.ssd_postprocess(cls_i, loc_i))
                self.metric.update(preds_batch, raw)
                
                current += self.cfg["train"]["batch_size"]
                if(current >= count):
                    break

        res = self.metric.compute()
        
        return  res["mAP"]

    def predict(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)

        self.model.eval()
        with torch.no_grad():
            loc_p, cls_p = self.model(img)
            
        result = self.post.ssd_postprocess(cls_p.squeeze(), loc_p.squeeze())
        return result