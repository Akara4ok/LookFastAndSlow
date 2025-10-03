import time
import logging
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from ObjectDetector.Anchors.anchors import Anchors, AnchorSpec
from ObjectDetector.Models.fast_and_slow_ssd import LookFastSlowSSD
from ObjectDetector.postprocessing import PostProcessor
from ObjectDetector.loss import SSDLoss
from ObjectDetector.map import MeanAveragePrecision
from ObjectDetector.phase2_loader import load_phase2_from_phase1

class VideoObjectDetector:
    def __init__(self, labels: list[str], config: dict, specs: list[AnchorSpec],
                 device: torch.device | str | None = None):
        self.cfg = config
        self.labels = labels
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        a_cfg = self.cfg["anchors"]
        self.anchors = Anchors(specs, self.cfg["model"]["img_size"], a_cfg["variances"], device=self.device)

        m_cfg = self.cfg["model"]

        self.model = LookFastSlowSSD(
            num_classes=len(labels),
            aspects=self.anchors.aspects,
            img_size=m_cfg["img_size"],
            fast_width=m_cfg["fast_width"],
            lstm_kernel=3,
            run_slow_every=4
        ).to(self.device)
        
        pretrain_path = m_cfg["pretrain"]
        if(pretrain_path is not None):
            self.model = load_phase2_from_phase1(self.model, pretrain_path, self.device)

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
        if min_lr is not None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg["train"]["epochs"],
                eta_min=min_lr
            )

    def check_dims(self):
        n_anchors = self.anchors.corner_anchors.size(0)
        T = self.cfg["model"].get("seq_len", 6)
        test_tensor = torch.zeros(2, T, 3, self.cfg["model"]["img_size"], self.cfg["model"]["img_size"], device=self.device)
        with torch.no_grad():
            loc_out, cls_out = self.model(test_tensor)
        if cls_out.size(2) != n_anchors:
            logging.warning(f"Model produces {cls_out.size(2)} anchors but generator made {n_anchors}")

    def _encode_step(self, img: torch.Tensor, target: dict) -> tuple[torch.Tensor, torch.Tensor]:
        boxes  = target["boxes"].to(self.device)
        labels = target["labels"].to(self.device)
        loc_t, cls_t = self.anchors.match(boxes, labels, self.cfg["anchors"]["iou_threshold"])
        return cls_t.to(self.device), loc_t.to(self.device)

    def _collate(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        x_seqs, cls_ts, loc_ts, raws = [], [], [], []

        for frames, targets in batch:
            T = len(frames)
            if isinstance(frames[0], torch.Tensor):
                x_t = torch.stack(frames, dim=0).to(self.device)
            else:
                fts = []
                for f in frames:
                    if isinstance(f, torch.Tensor):
                        ft = f
                    else:
                        ft = torch.from_numpy(f).permute(2, 0, 1).float()
                    fts.append(ft)
                x_t = torch.stack(fts, dim=0).to(self.device)

            cls_list, loc_list = [], []
            for t in range(T):
                cls_t, loc_t = self._encode_step(x_t[t], targets[t])
                cls_list.append(cls_t)
                loc_list.append(loc_t)
            cls_ts.append(torch.stack(cls_list, dim=0))
            loc_ts.append(torch.stack(loc_list, dim=0))
            x_seqs.append(x_t)
            raws.append(targets)

        x_batch   = torch.stack(x_seqs, dim=0).to(self.device)
        cls_batch = torch.stack(cls_ts,  dim=0).to(self.device)
        loc_batch = torch.stack(loc_ts,  dim=0).to(self.device)
        return x_batch, cls_batch, loc_batch, raws

    def _split_datasets(self, full_ds: Dataset, test_ratio: float):
        test_len = int(len(full_ds) * test_ratio)
        train_len = len(full_ds) - test_len
        train_ds, val_ds = random_split(full_ds, [train_len, test_len])
        return train_ds, val_ds

    def train(self, ds: Dataset):
        logging.info("Training started (video)")
        ds_train, ds_val = self._split_datasets(ds, self.cfg["data"]["test_percent"])

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg["train"]["batch_size"],
            shuffle=True,
            collate_fn=self._collate,
            drop_last=True,
            num_workers=self.cfg["data"].get("num_workers", 0),
            pin_memory=True
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=self.cfg["train"]["batch_size"],
            shuffle=False,
            collate_fn=self._collate,
            drop_last=False,
            num_workers=0
        )

        best_map = -1.0
        ckpt_dir = Path(self.cfg["model"]["path"]).expanduser().parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg["train"]["epochs"]):
            compute_metric = (epoch % 5 == 0)
            t0 = time.time()
            tr = self._run_epoch(dl_train, True, compute_metric)
            vl = self._run_epoch(dl_val,   False, True)
            dt = time.time() - t0

            (train_loss, train_loc, train_cls, train_map) = tr
            (val_loss,   val_loc,   val_cls,   val_map)   = vl

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/train_loc", train_loc, epoch)
            self.writer.add_scalar("loss/train_cls", train_cls, epoch)
            self.writer.add_scalar("loss/train_map", train_map, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("loss/val_loc", val_loc, epoch)
            self.writer.add_scalar("loss/val_cls", val_cls, epoch)
            self.writer.add_scalar("loss/val_map", val_map, epoch)

            if self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Epoch {epoch:03d} time {dt:.1f}s lr {lr:.6f}")
            else:
                logging.info(f"Epoch {epoch:03d} time {dt:.1f}s")

            msg = f"train {train_loss:.4f} loc {train_loc:.4f} cls {train_cls:.4f}"
            if compute_metric:
                msg += f" map {train_map:.4f}"
            logging.info(msg)
            logging.info(f"val   {val_loss:.4f} loc {val_loc:.4f} cls {val_cls:.4f} map {val_map:.4f}")
            logging.info("=" * 30)

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
        total_loss, total_loc, total_cls, n_batches = 0.0, 0.0, 0.0, 0

        with torch.set_grad_enabled(train):
            loop = tqdm(loader, desc="Train" if train else "Val")
            for batch in loop:
                if batch is None:
                    continue

                x_seq, cls_gt, loc_gt, raw = batch
                B, T = x_seq.size(0), x_seq.size(1)

                if train:
                    self.optimizer.zero_grad()

                loc_p, cls_p = self.model(x_seq)

                loc_loss_sum, cls_loss_sum = 0.0, 0.0
                for t in range(T):
                    l_loc, l_cls = self.criterion(loc_p[:, t], cls_p[:, t], loc_gt[:, t], cls_gt[:, t])
                    loc_loss_sum += l_loc
                    cls_loss_sum += l_cls
                loc_loss = loc_loss_sum / T
                cls_loss = cls_loss_sum / T
                loss = loc_loss + cls_loss

                if compute_metric:
                    preds_batch = []
                    for b in range(B):
                        preds_batch.append(self.post.ssd_postprocess(cls_p[b, -1], loc_p[b, -1]))
                    last_raw = [r[-1] for r in raw]
                    self.metric.update(preds_batch, last_raw)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()

                total_loss += float(loss.item())
                total_loc  += float(loc_loss.item())
                total_cls  += float(cls_loss.item())
                n_batches  += 1

        res = {"mAP": 0.0}
        if compute_metric:
            res = self.metric.compute()

        return (total_loss / max(1, n_batches),
                total_loc  / max(1, n_batches),
                total_cls  / max(1, n_batches),
                res["mAP"])

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, frames: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            x = torch.stack(frames, dim=0).unsqueeze(0).to(self.device)  # (1,T,3,H,W)
            loc_p, cls_p = self.model(x)
            result = self.post.ssd_postprocess(cls_p[0, -1], loc_p[0, -1])
        return result
