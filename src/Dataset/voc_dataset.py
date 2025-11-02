import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Literal
import hashlib

from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

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
                 use_cache = True):
        super().__init__()
        self.root = Path(root).expanduser()
        self.year = str(year)
        self.split = split

        self._download_if_needed()
        self.image_ids = self._read_split_file()

        self.img_dir = self.root / f"VOC{self.year}" / "JPEGImages"
        self.ann_dir = self.root / f"VOC{self.year}" / "Annotations"

        self.class_to_idx = {name: i for i, name in enumerate(self.VOC_CLASSES)}
        
        self.use_cache = use_cache
        self.is_cached = False
        self.cached_data = []
        
        if(self.use_cache):
            self._build_cache()

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

        boxes, labels = [], []
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

        return np.array(boxes), np.array(labels)


    def _build_cache(self):
        for i in range(len(self.image_ids)):
            self.__getitem__(i)
        self.is_cached = True

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        if(self.use_cache and self.is_cached):
            img, tgt = self.cached_data[idx]
            return img.copy(), {"boxes": tgt["boxes"].copy(), "labels": tgt["labels"].copy()}

        img_id = self.image_ids[idx]
        img_path = self.img_dir / f"{img_id}.jpg"
        ann_path = self.ann_dir / f"{img_id}.xml"

        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)

        boxes, labels = self._parse_annotation(ann_path)
        target = {"boxes": boxes, "labels": labels}
        
        if(self.use_cache):
            self.cached_data.append((img.copy(),
                             {"boxes": target["boxes"].copy(),
                              "labels": target["labels"].copy()}))

        return img, target