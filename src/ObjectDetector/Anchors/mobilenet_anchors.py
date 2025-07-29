from ObjectDetector.Anchors.anchors import AnchorSpec, AnchorSizeRange

specs = [
    AnchorSpec(19, 16, AnchorSizeRange(20, 45),  [2]),
    AnchorSpec(10, 32, AnchorSizeRange(45, 90),  [2]),
    AnchorSpec(5,  64, AnchorSizeRange(90, 135), [2]),
    AnchorSpec(3, 100, AnchorSizeRange(135,180), [2]),
    AnchorSpec(2, 150, AnchorSizeRange(180,225), [2]),
    AnchorSpec(1, 300, AnchorSizeRange(225,270), [2]),
]