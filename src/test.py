import torch
from ObjectDetector.Anchors.mobilenet_anchors import specs
from ObjectDetector.Anchors.anchors import Anchors

anchors = Anchors(specs, 300, [0.1, 0.1, 0.2, 0.2])
print(anchors.center_anchors.shape)

