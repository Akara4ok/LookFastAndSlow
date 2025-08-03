import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Literal
import hashlib

import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as TF

class VOCDataset(Dataset):
    _URLS = {
        "2007": {
            "trainval": ("http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
                         "VOCtrainval_06-Nov-2007.tar",
                         "c52e279531787c972589f7e41ab4ae64"),
            "test":     ("http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
                         "VOCtest_06-Nov-2007.tar",
                         "b6e924de25625d8de591ea690078ad9f"),
        },
        "2012": {
            "trainval": ("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
                         "VOCtrainval_11-May-2012.tar",
                         "6cd6e144f989b92b3379bac3b3de84fd"),
        },
    }
    
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    ]  

    def __init__(self,
                 root: str | Path,
                 year: int,
                 split: Literal["train", "val", "trainval", "test"] = "trainval",
                 img_size: int = 300):
        super().__init__()
        self.root = Path(root).expanduser()
        self.year = str(year)
        self.split = split
        self.img_size = img_size

        self._download_if_needed()
        self.image_ids = self._read_split_file()

        self.img_dir = self.root / f"VOC{self.year}" / "JPEGImages"
        self.ann_dir = self.root / f"VOC{self.year}" / "Annotations"

        self.class_to_idx = {name: i + 1 for i, name in enumerate(self.VOC_CLASSES)}

    def _download_if_needed(self):
        needed_keys = ["trainval"] if self.split != "test" else ["test"]
        for k in needed_keys:
            url, md5 = self._URLS[self.year][k]
            tar_name = Path(url).name
            target   = self.root / tar_name

            voc_folder = self.root / f"VOC{self.year}"
            if voc_folder.exists():
                continue

            if not target.exists():
                logging.info(f"Downloading {tar_name} …")
                self.root.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, target)

            if self._md5(target) != md5:
                target.unlink(missing_ok=True)
                raise RuntimeError("MD5 mismatch. Deleted corrupted archive.")

            logging.info(f"Extracting {tar_name} …")
            with tarfile.open(target) as tar:
                tar.extractall(path=self.root)
            target.unlink()

    @staticmethod
    def _md5(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.md5()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

            
    def _read_split_file(self) -> List[str]:
        list_file = (self.root / f"VOC{self.year}" /
                     "ImageSets" / "Main" / f"{self.split}.txt")
        return [l.strip() for l in list_file.read_text().splitlines()]
    
    def _parse_annotation(self, xml_path: Path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)

        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name not in self.class_to_idx:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) / w
            ymin = float(bbox.find("ymin").text) / h
            xmax = float(bbox.find("xmax").text) / w
            ymax = float(bbox.find("ymax").text) / h
            boxes.append([ymin, xmin, ymax, xmax])
            labels.append(self.class_to_idx[name])

        return torch.tensor(boxes, dtype=torch.float32), \
               torch.tensor(labels, dtype=torch.int64)


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_path = self.img_dir / f"{img_id}.jpg"
        ann_path = self.ann_dir / f"{img_id}.xml"

        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, (self.img_size, self.img_size))
        img = TF.to_tensor(img)

        boxes, labels = self._parse_annotation(ann_path)
        target = {"boxes": boxes, "labels": labels}

        return img, target