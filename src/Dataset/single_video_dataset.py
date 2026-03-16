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

    def __init__(self, root: str | Path, seq_len: int = 6):

        self.video_path = Path(root + "/video.mp4")
        self.root = Path(root)

        self.class_to_idx = {c: i for i, c in enumerate(self.VOC_CLASSES)}

        self.image_ids = self._read_split_file()

        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.seq_len = seq_len

    def _read_split_file(self):
        txt_path = self.root / "ImageSets" / "Main" / f"Train.txt"
        return [l.strip() for l in txt_path.read_text().splitlines()]


    def _parse_annotation(self, xml_path: Path):
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

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


    def _read_frame(self, frame_num: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_num}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame.astype(np.float32)

    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):
        frame_idx = self.seq_len * idx
        frames = []
        targets = []

        for i in range(self.seq_len):
            frame_id_str = self.image_ids[frame_idx + i]

            frame_num = int(frame_id_str.replace("frame_", ""))  # e.g. "000123" → 123
            frames.append(self._read_frame(frame_num))

            xml_path = self.root / "Annotations" / f"{frame_id_str}.xml"
            boxes, labels = self._parse_annotation(xml_path)

            targets.append({"boxes": boxes, "labels": labels})

        return frames, targets