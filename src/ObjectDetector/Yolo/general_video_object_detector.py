import logging
from typing import Dict
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO

from ObjectDetector.map import MeanAveragePrecision
from ObjectDetector.Yolo.general_image_object_detector import GeneralImageObjectDetector
from Dataset.Yolo.YoloSegDataset import YoloSeqTestDataset, InferenceTransform
from ObjectDetector.Yolo.Models.yolo_fast_and_slow import YoloFastAndSlow

class GeneralVideoObjectDetector(GeneralImageObjectDetector):
    def __init__(self, config: Dict, labels = None, map_classes = None, device: torch.device | str | None = None):
        super().__init__(config, labels, map_classes, device)
        self.model: nn.Module = None
        self.inference_transform = InferenceTransform(self.config["model"]["img_size"])

    def set_nms_params(self, iou: float, conf: float):
        self.model.set_nms_params(iou, conf)

    def load_weights(self, weights_path: str):
        full_model = torch.load(weights_path, map_location="cuda", weights_only=False)
        self.model = full_model["model"]
        self.model.load_state_dict(full_model["state_dict"])
        self.model.eval().to("cuda")

    def remap_from_letterbox(self,
                         boxes_xyxy: np.ndarray,
                         original_size: tuple[int, int],
                         padded_size: int,
                         scale: float,
                         pad: tuple[int, int]):
        h0, w0 = original_size
        pad_left, pad_top = pad

        inv_scale_w = 1.0 / (scale * w0)
        inv_scale_h = 1.0 / (scale * h0)
        pad_left_norm = pad_left / padded_size
        pad_top_norm  = pad_top  / padded_size

        out = boxes_xyxy

        out[:, [0, 2]] = (out[:, [0, 2]] - pad_left_norm) * padded_size * inv_scale_w

        out[:, [1, 3]] = (out[:, [1, 3]] - pad_top_norm) * padded_size * inv_scale_h

        np.clip(out, 0.0, 1.0, out=out)

        return out.astype(np.float32)


    def postprocess_single(
        self,
        res: np.ndarray,
        return_boxes_to_original: bool = False,
        original_size: tuple[int, int] | None = None,
        letterbox_params=None,
    ) -> dict:
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "classes": np.zeros((0,), np.int64)}

        boxes_xyxy = r0.boxes.xyxyn.detach().cpu().numpy().astype(np.float32)
        scores = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = r0.boxes.cls.detach().cpu().numpy().astype(np.int64)

        if return_boxes_to_original:
            boxes_xyxy = self.remap_from_letterbox(
                boxes_xyxy,
                original_size=original_size,
                padded_size=self.config["model"]["img_size"],
                scale=letterbox_params["scale"],
                pad=letterbox_params["pad"],
            )

        predicted = {"boxes": boxes_xyxy, "scores": scores, "classes": classes}
        return self.map_result_classes(predicted)

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        original_size = frame.shape[:2]
        lb_params = {}
        if isinstance(frame, np.ndarray):
            frame, lb_params = self.inference_transform(frame)
            frame = frame.unsqueeze(0)
        res = self.model.predict(frame)
        return self.postprocess_single(res, True, original_size, lb_params)
    
    @torch.no_grad()
    def predict_seq(self, batch: np.ndarray) -> list[dict]:
        res_frames = self.model.predict(batch)

        results = []
        for res in res_frames:
            results.append(self.postprocess_single(res))
        
        return results

    def test(self, ds: Dataset, count: int) -> float:
        logging.info("Testing started")
        ds = YoloSeqTestDataset(ds, self.config["model"]["img_size"])
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
                imgs = batch["images"].to(self.device)

                preds_batch = self.predict_seq(imgs)

                for preds, raw in zip(preds_batch, batch["raw"][0]):
                    tensor_dict = {k: torch.from_numpy(v) for k, v in preds.items()}
                    metric.update([tensor_dict], [raw])
                
                current += self.config["train"]["batch_size"]
                if(current >= count):
                    break

        res = metric.compute()
        
        # return  res["weighted_mAP"]
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
    

    def test_collate(self, batch):
        imgs = [img_seq for img_seq, _ in batch] # list of (T,C,H,W)
        imgs = torch.stack(imgs, dim=0) # (B,T,C,H,W)

        boxes_batch = []
        labels_batch = []
        raw_batch = []

        for _, tgts in batch:
            seq_boxes, seq_labels, seq_raw = [], [], []
            for frame_tgt in tgts:
                seq_boxes.append(frame_tgt["boxes"])
                seq_labels.append(frame_tgt["labels"])
                seq_raw.append(frame_tgt)
            boxes_batch.append(seq_boxes)
            labels_batch.append(seq_labels)
            raw_batch.append(seq_raw)

        return {
            "images": imgs,
            "boxes": boxes_batch,
            "labels": labels_batch,
            "raw": raw_batch,
        }
    