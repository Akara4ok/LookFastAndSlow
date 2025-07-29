from ObjectDetector.Anchors.anchors import AnchorSpec, AnchorSizeRange

specs = [
    AnchorSpec(19, 16, AnchorSizeRange(60, 105),  [2, 3]),
    AnchorSpec(10, 32, AnchorSizeRange(105, 150), [2, 3]),
    AnchorSpec(5,  64, AnchorSizeRange(150, 195), [2, 3]),
    AnchorSpec(3, 100, AnchorSizeRange(195, 240), [2, 3]),
    AnchorSpec(2, 150, AnchorSizeRange(240, 285), [2, 3]),
    AnchorSpec(1, 300, AnchorSizeRange(285, 330), [2, 3]),
]