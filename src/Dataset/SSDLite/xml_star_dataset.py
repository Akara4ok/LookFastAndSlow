import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class XMLStarDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 img_size: Tuple[int, int]):
        super().__init__()
        self.dataset_path = dataset_path
        self.img_size = img_size

        self.samples, self.label_map = self._scan_dirs()
        self.labels = [name for name, _ in sorted(self.label_map.items(),
                                                  key=lambda p: p[1])]

    @staticmethod
    def _parse_xml(xml_path: str) -> Tuple[str, List[List[float]], List[str]]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        size = root.find("size")

        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            labels.append(obj.find("name").text)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        return filename, boxes, labels

    def _scan_dirs(self) -> Tuple[List[Dict], Dict[str, int]]:
        annotation_dir = self.dataset_path + "/annotations"
        images_dir = self.dataset_path + "/images"

        samples, label_map = [], {}
        next_id = 0

        for xml_file in os.listdir(annotation_dir):
            if not xml_file.endswith(".xml"):
                continue

            xml_path = os.path.join(annotation_dir, xml_file)
            filename, boxes, labels = self._parse_xml(xml_path)

            label_ids  = []
            for label in labels:
                if label not in label_map:
                    label_map[label] = next_id
                    next_id += 1
                label_ids .append(label_map[label])

            samples.append({
                "image_path": os.path.join(images_dir, filename),
                "boxes": boxes,
                "labels": label_ids
            })

        return samples, label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = np.array(Image.open(sample["image_path"]).convert("RGB")).astype(np.float32) / 255

        boxes  = np.array(sample["boxes"])
        labels = np.array(sample["labels"])

        target = {"boxes": boxes, "labels": labels}
        return img, target
