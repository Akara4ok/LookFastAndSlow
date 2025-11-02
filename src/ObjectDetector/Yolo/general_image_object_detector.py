import logging
from typing import Dict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO

from ObjectDetector.map import MeanAveragePrecision
from Dataset.Yolo.YoloTestDataset import YoloTestDataset

class GeneralImageObjectDetector:
    def __init__(self, config: Dict, labels = None, map_classes = None, device: torch.device | str | None = None):
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model: YOLO = None
        self.map_classes = map_classes
        self.labels = labels

    def load_weights(self, weights_path: str):
        self.model = YOLO(weights_path)
        if self.labels is None:
            self.labels = self.model.names
        self.model.to(self.device)
        logging.info(f"Weights loaded from {weights_path} to device {self.device}")

    def train(self, data_yaml_path: str):
        logging.info("Training started")
        self.model.train(
            data=data_yaml_path,
            epochs=self.config["train"]["epochs"],
            imgsz=self.config["model"]["img_size"],
            batch=self.config["train"]["batch_size"],
            workers=4,
            device=0,
            patience=20,
            optimizer="auto",
            lr0=self.config["lr"]["initial_lr"],
            weight_decay=0.0005,
            project="Model/Yolo",
            name="YoloVoc",
            verbose=True,
            pretrained=True,
            exist_ok=True
        )

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

        metric = MeanAveragePrecision(num_classes=len(self.model.names), device=self.device)
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
    
    def test_collate(self, batch):
        imgs, boxes, labels, raw = [], [], [], []
        for img, tgt in batch:
            imgs.append(img)
            boxes.append(torch.as_tensor(tgt["boxes"], dtype=torch.float32))
            labels.append(torch.as_tensor(tgt["labels"], dtype=torch.float32))
            raw.append(tgt)

        images = torch.stack(imgs, dim=0)  # [B,C,H,W]
        return {"images": images, "boxes": boxes, "labels": labels, "raw": raw}
    