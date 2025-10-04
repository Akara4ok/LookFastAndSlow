
import os
import json
import shutil
import logging
import hashlib
import pathlib
import urllib.request
from typing import List, Optional, Dict, Tuple
import random
import numpy as np
import bisect

import cv2
import pandas as pd
from torch.utils.data import Dataset

class QuietLogger:
    def debug(self, msg):   pass
    def warning(self, msg): pass
    def error(self, msg):   pass 

YTBB_BASE = "https://research.google.com/youtube-bb"
CSV_URLS = {
    "train": f"{YTBB_BASE}/yt_bb_detection_train.csv.gz",
    "val":   f"{YTBB_BASE}/yt_bb_detection_validation.csv.gz",
}

def _safe_mkdir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
def _download(url: str, dst: str) -> bool:
    try:
        urllib.request.urlretrieve(url, dst)  # nosec - user-requested download
        return True
    except Exception as e:
        return False
    
def _download_video_ytdlp(youtube_id: str, out_dir: str, retries: int = 3) -> Optional[str]:
    try:
        import yt_dlp
    except Exception:
        logging.error("yt_dlp not installed. pip install yt-dlp")
        return None

    _safe_mkdir(out_dir)

    fmt = (
        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/"
        "best[ext=mp4][height<=720]/"
        "bestvideo[ext=webm][height<=720]/"
        "best"
    )

    outtmpl = os.path.join(out_dir, "%(id)s.%(ext)s")

    # If already downloaded, return it
    for ext in (".mp4", ".webm", ".mkv", ".mov"):
        p = os.path.join(out_dir, youtube_id + ext)
        if os.path.exists(p):
            return p

    ydl_opts = {
        "logger:": QuietLogger(),
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": retries,
        "format": fmt,
        "postprocessors": [],
        "merge_output_format": None,
        "continuedl": True,
        "ignoreerrors": True,
    }

    url = f"https://youtu.be/{youtube_id}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ext = (info.get("ext") or "mp4")
            candidate = os.path.join(out_dir, f"{youtube_id}.{ext}")
            if os.path.exists(candidate):
                return candidate

            reqs = info.get("requested_downloads") or []
            for rd in reqs:
                ext2 = rd.get("ext") or ext
                cand2 = os.path.join(out_dir, f"{youtube_id}.{ext2}")
                if os.path.exists(cand2):
                    return cand2

            # Fallback: any file starting with id.
            for fn in os.listdir(out_dir):
                if fn.startswith(youtube_id + "."):
                    return os.path.join(out_dir, fn)
    except Exception as e:
        return None

    return None

class YoutubeBBDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        frames_per_clip: int = 6,
        download: bool = True,
        cache_index: bool = True,
    ):
        assert split in ("train", "val")
        self.root = os.path.abspath(root)
        self.split = split
        self.frames_per_clip = int(frames_per_clip)
        self.download = download
        self.video_dir = os.path.join(self.root, "videos")
        self.ann_dir = os.path.join(self.root, "annotations")
        self.cache_index = cache_index

        if not os.path.exists(self.root):
            if not self.download:
                raise FileNotFoundError(
                    f"Root folder {self.root} does not exist and download=False."
                )
            _safe_mkdir(self.root)

        _safe_mkdir(self.ann_dir)
        _safe_mkdir(self.video_dir)

        csv_gz_path = os.path.join(self.ann_dir, f"yt_bb_detection_{'train' if split=='train' else 'validation'}.csv.gz")
        if (not os.path.exists(csv_gz_path)) and self.download:
            url = CSV_URLS["train" if split == "train" else "val"]
            logging.info(f"Downloading YT-BB annotations: {url}")
            _download(url, csv_gz_path)

        if not os.path.exists(csv_gz_path):
            raise FileNotFoundError(
                f"Missing annotation file {csv_gz_path}. Set download=True to fetch."
            )

        self.index_path = os.path.join(self.ann_dir, f"segments_{split}.json")
        if self.cache_index and os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.segments = json.load(f)
        else:
            self.segments = self._build_segments(csv_gz_path)
            if self.cache_index:
                with open(self.index_path, "w", encoding="utf-8") as f:
                    json.dump(self.segments, f)

        if len(self.segments) == 0:
            raise RuntimeError("No usable YT-BB segments found after filtering.")
        
    def _build_segments(self, csv_gz_path: str) -> List[Dict]:
        logging.info(f"Parsing {csv_gz_path}")
        df = pd.read_csv(csv_gz_path, compression="gzip", header=None, names=[
            "youtube_id", "timestamp_ms", "class_id", "class_name",
            "object_id", "object_presence", "xmin", "xmax", "ymin", "ymax"
        ])
        df = df[df["object_presence"] == "present"]
        
        grouped = df.groupby(["youtube_id", "class_id", "object_id"])
        segments: List[Dict] = []
        for (youtube_id, class_id, object_id), group in grouped:
            group_sorted = group.sort_values("timestamp_ms")
            timestamps_ms = group_sorted["timestamp_ms"].astype(int).tolist()
            bboxes = group_sorted[["xmin", "xmax", "ymin", "ymax"]].astype(float).values.tolist()
            labels = group_sorted["class_id"].astype(int).tolist()

            segments.append({
                "youtube_id": str(youtube_id),
                "class_id": int(class_id),
                "object_id": int(object_id),
                "timestamps_ms": timestamps_ms,
                "bboxes": bboxes,
                "labels": labels,
            })
            
        return segments
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def _ensure_downloaded(self, segment_id: int):
        seg = self.segments[segment_id]
        youtube_id   = seg["youtube_id"]
        return self._ensure_video(youtube_id)
    
    def _ensure_video(self, youtube_id: str) -> Optional[str]:
        for ext in (".mp4", ".webm", ".mkv", ".mov"):
            p = os.path.join(self.video_dir, youtube_id + ext)
            if os.path.exists(p):
                return p
        if not self.download:
            return None
        return _download_video_ytdlp(youtube_id, self.video_dir)
    
    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        youtube_id   = seg["youtube_id"]
        ts_list      = seg["timestamps_ms"]
        bboxes_list  = seg["bboxes"]
        labels_list  = seg["labels"]

        # Ensure sorted and build a quick lookup: ts -> (bbox, label)
        pairs = sorted(zip(ts_list, bboxes_list, labels_list), key=lambda t: int(t[0]))
        ann_ts_sorted = [int(t) for t, _, _ in pairs]
        ann_map = {int(t): (bbox, int(lbl)) for t, bbox, lbl in pairs}

        # ------- open video -------
        video_path = self._ensure_video(youtube_id)
        if video_path is None or (not os.path.exists(video_path)):
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        T = int(self.frames_per_clip)

        stride = max(1, random.randint(0, 3))
        match_window_ms = int(getattr(self, "annotation_match_window_ms", 500))

        if not ann_ts_sorted:
            cap.release()
            return None
        
        t0 = random.choice(ann_ts_sorted)

        jitter_frames = random.randint(0, max(1, int(0.2 * fps)))  # ~0–0.2s back
        start_idx = max(0, int(round((t0 / 1000.0) * fps)) - jitter_frames)

        max_start = max(0, total_frames - (T - 1) * stride - 1)
        start_idx = min(start_idx, max_start)

        frames: List[np.ndarray] = []
        targets: List[Dict[str, np.ndarray]] = []

        def nearest_ts(ms: int) -> Optional[int]:
            i = bisect.bisect_left(ann_ts_sorted, ms)
            cand = []
            if i < len(ann_ts_sorted): cand.append(ann_ts_sorted[i])
            if i > 0:                  cand.append(ann_ts_sorted[i - 1])
            if not cand: return None
            best = min(cand, key=lambda t: abs(t - ms))
            return best if abs(best - ms) <= match_window_ms else None

        fi = start_idx
        tries = 0
        max_tries = 5 * (T + 10)

        while len(frames) < T and tries < max_tries:
            tries += 1
            if fi >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok or img is None:
                fi += stride
                continue

            # frame timestamp in ms
            t_ms = int(round((fi / max(1e-6, fps)) * 1000.0))
            t_match = nearest_ts(t_ms)
            if t_match is None:
                # no bbox near this raw frame — skip it
                fi += stride
                continue

            # fetch bbox + label for this object at matched timestamp
            bbox_norm, cls_id = ann_map[int(t_match)]
            H, W = img.shape[:2]
            # normalize -> absolute xyxy
            xmin, xmax, ymin, ymax = [float(v) for v in bbox_norm]
            x1 = max(0.0, min(W - 1.0, xmin * W))
            x2 = max(0.0, min(W - 1.0, xmax * W))
            y1 = max(0.0, min(H - 1.0, ymin * H))
            y2 = max(0.0, min(H - 1.0, ymax * H))
            box_xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            labels_np = np.array([int(cls_id)], dtype=np.int64)

            frames.append(img)
            targets.append({"boxes": box_xyxy, "labels": labels_np})

            fi += stride

        cap.release()

        if len(frames) < T:
            return None
            
        return frames, targets

    