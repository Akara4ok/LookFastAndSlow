import tensorflow as tf
from tensorflow.keras import layers, models, activations
from ObjectDetector.Models.backbone import Backbone

class SSDHead():
    def __init__(self, backbone: Backbone, num_labels: int, aspects: list[list[float]]):
        self.loc_final, self.cls_final = self.build_head(backbone, num_labels, aspects)
        pass
    
    def concatenate_head(self, x: tf.Tensor, dim: int, name: str) -> tf.Tensor:
        reshaped = [layers.Reshape((-1, dim))(layer) for layer in x]
        return layers.Concatenate(axis=1, name=name)(reshaped)

    def build_head(self, backbone: Backbone, num_labels: int, aspects: list[list[float]]) -> tuple[tf.Tensor, tf.Tensor]:
        aspects_size = [len(x) + 1 for x in aspects]
        labels = []
        boxes = []
        for i, output in enumerate(backbone.bridge_layers):
            aspect_size = aspects_size[i]
            labels.append(layers.SeparableConv2D(aspect_size * num_labels, (3, 3), padding="same", activation="softmax")(output))
            boxes.append(layers.SeparableConv2D(aspect_size * 4, (3, 3), padding="same")(output))
        pred_labels = self.concatenate_head(labels, num_labels, name="cls")
        pred_deltas = self.concatenate_head(boxes, 4, name="loc")
        return pred_deltas, pred_labels
