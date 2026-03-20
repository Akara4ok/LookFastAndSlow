import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


class SingleVideoDataset(Dataset):
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    ]

    def __init__(
        self,
        root: str | Path,
        seq_len: int = 6,
        cache_frames: bool = True,
        cache_annotations: bool = True,
    ):
        self.root = Path(root)
        self.video_path = self.root / "video.mp4"
        self.seq_len = seq_len

        self.class_to_idx = {c: i for i, c in enumerate(self.VOC_CLASSES)}
        self.image_ids = self._read_split_file()

        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cache_frames = cache_frames
        self.cache_annotations = cache_annotations

        self.frame_cache = {}
        self.annotation_cache = {}

    def _read_split_file(self):
        txt_path = self.root / "ImageSets" / "Main" / "Train.txt"
        return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]

    def _parse_annotation(self, xml_path: Path):
        if self.cache_annotations and xml_path in self.annotation_cache:
            return self.annotation_cache[xml_path]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name not in self.class_to_idx:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.cache_annotations:
            self.annotation_cache[xml_path] = (boxes, labels)

        return boxes, labels

    def _read_frame(self, frame_num: int):
        if self.cache_frames and frame_num in self.frame_cache:
            return self.frame_cache[frame_num]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_num}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)

        if self.cache_frames:
            self.frame_cache[frame_num] = frame

        return frame

    def __len__(self):
        return len(self.image_ids) // self.seq_len

    def __getitem__(self, idx):
        frame_idx = self.seq_len * idx
        frames = []
        targets = []

        for i in range(self.seq_len):
            frame_id_str = self.image_ids[frame_idx + i]
            frame_num = int(frame_id_str.replace("frame_", ""))

            frames.append(self._read_frame(frame_num))

            xml_path = self.root / "Annotations" / f"{frame_id_str}.xml"
            boxes, labels = self._parse_annotation(xml_path)

            targets.append({
                "boxes": boxes,
                "labels": labels
            })

        return frames, targets