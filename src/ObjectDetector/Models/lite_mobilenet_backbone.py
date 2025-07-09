import tensorflow as tf
from tensorflow.keras import layers, models, activations
from ObjectDetector.Models.backbone import Backbone

class LiteMobileNetBackbone(Backbone):
    def __init__(self, inputs: tf.Tensor):
        super().__init__()
        self.model, self.bridge_layers = self.build_model(inputs)
    
    def build_model(self, inputs: tf.Tensor) -> tuple[tf.keras.Model, list[layers.Layer]]:
        base_model = tf.keras.applications.MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
        base_model.trainable = False
        
        first_layer = base_model.get_layer("block_13_expand_relu").output
        second_layer = base_model.output
        
        extra1 = layers.Conv2D(256, (1, 1), activation="relu")(second_layer)
        extra1 = layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same", activation="relu")(extra1)

        extra2 = layers.Conv2D(128, (1, 1), activation="relu")(extra1)
        extra2 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(extra2)

        extra3 = layers.Conv2D(128, (1, 1), activation="relu")(extra2)
        extra3 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(extra3)

        extra4 = layers.Conv2D(128, (1, 1), activation="relu")(extra3)
        extra4 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(extra4)
        
        return base_model, [first_layer, second_layer, extra1, extra2, extra3, extra4]
    