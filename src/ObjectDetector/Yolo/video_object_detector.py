from typing import Optional
from ultralytics import YOLO
import torch
import numpy as np
from Dataset.SSDLite.test_dataset import TestDataset
from torch.utils.data import DataLoader
from ObjectDetector.map import MeanAveragePrecision

MAP_CLASS = {
    0: 15,
    1: 2,
    2: 7,
    3: 14,
    4: 1,
    5: 6,
    6: 19,
    7: 7,
    8: 4,
    14: 3,
    15: 8,
    16: 12,
    17: 13,
    18: 17,
    19: 10,
    39: 5,
    56: 9,
    57: 18,
    58: 16,
    60: 11,
    62: 20
}

class VideoObjectDetector:
    def __init__(
        self,
        labels,
        config,
        specs,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.labels = labels
        self.metric = MeanAveragePrecision(num_classes=len(self.labels), device=self.device)
        self.cfg = config
    
    def load_weights(self, path):
        self.model = YOLO(path)
        self.model.to(self.device)
        
    def predict(self, frame):
        results = self.model.predict(frame, verbose=False)[0]
        return self._results_to_pred_dict(results)
    
    def _results_to_pred_dict(self, r):
        boxes = r.boxes.xyxyn.cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        classes = r.boxes.cls.cpu().numpy().astype(np.int64)

        mapped_boxes = []
        mapped_scores = []
        mapped_classes = []


        for box, score, cls_id in zip(boxes, scores, classes):
            print(box)
            cid = int(cls_id.item())
            if cid in MAP_CLASS:
                mapped_boxes.append(np.expand_dims(box, axis=0))
                mapped_scores.append(np.expand_dims(score, axis=0))
                mapped_classes.append(MAP_CLASS[cid])

        if len(mapped_boxes) > 0:
            boxes_out = np.concatenate(mapped_boxes, axis=0)
            scores_out = np.concatenate(mapped_scores, axis=0)
            classes_out = np.stack(mapped_classes, axis=0)
        else:
            boxes_out = np.empty((0, 4), dtype=np.float32)
            scores_out = np.empty((0,), dtype=np.float32)
            classes_out = np.empty((0,), dtype=np.int64)

        return dict(boxes=boxes_out, scores=scores_out, classes=classes_out)
        
    def test(self, ds, count: int) -> float:
        ds = TestDataset(ds, self.cfg["model"]["img_size"])
        dl_test = DataLoader(
            ds,
            batch_size=self.cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate,
            drop_last=False,
        )

        test_map = self._test_model(dl_test, count)
        return test_map

    def _test_model(self, dl_test: DataLoader, count: int) -> float:
        self.model.model.eval()

        self.metric.reset()
        current = 0

        with torch.no_grad():
            for imgs, cls_gt, loc_gt, raw in dl_test:
                imgs_list = imgs
                if isinstance(imgs, torch.Tensor):
                    imgs = imgs.to(self.device, non_blocking=True)
                    imgs_list = self._prep_for_yolo_np(imgs)
                               

                results = self.model.predict(
                    imgs_list,
                    verbose=False,
                    device=self.device,
                )

                preds_batch = []
                for r in results:
                    preds_batch.append(self._results_to_pred_dict(r))
                    
                self.metric.update(preds_batch, raw)

                bsz = len(results)
                current += bsz
                if current >= count:
                    break

        res = self.metric.compute()
        return float(res["mAP"])

    def _prep_for_yolo_np(self, imgs: torch.Tensor) -> list:
        if imgs.dtype != torch.float32:
            imgs = imgs.float()

        imgs_list_np = []
        with torch.no_grad():
            for img in imgs:  # im: (C,H,W)
                hwc = img.permute(1, 2, 0).clamp(0, 1).numpy() * 255
                hwc = hwc.astype(np.uint8)
                imgs_list_np.append(hwc)
        return imgs_list_np


    def _collate(self, batch):
        imgs, cls_t, loc_t, raw = [], [], [], []
        for img, tgt in batch:
            imgs.append(img)
            cls_t.append(torch.Tensor())
            loc_t.append(torch.Tensor())
            raw.append(tgt)

        return (torch.stack(imgs).to(self.device), torch.stack(cls_t).to(self.device), torch.stack(loc_t).to(self.device), raw)
