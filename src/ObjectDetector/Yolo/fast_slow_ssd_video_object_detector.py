from typing import Dict, List

import torch

from ObjectDetector.Yolo.general_video_object_detector import GeneralVideoObjectDetector
from ObjectDetector.SSDLite.video_object_detector import VideoObjectDetector


class FastSlowSSDVideoObjectDetector(GeneralVideoObjectDetector):
    def __init__(self, config: Dict, labels: List[str], inference = False, device: torch.device | str | None = None):
        super().__init__(config, labels, None, device)
        self.model: VideoObjectDetector = None
        self.inference = inference

    def load_weights(self, weights_path: str):
        self.model = VideoObjectDetector(self.labels, self.config, [], self.device)
        self.model.load_weights(weights_path)

    