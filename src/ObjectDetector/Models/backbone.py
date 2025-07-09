import tensorflow as tf
from tensorflow.keras import layers

class Backbone():
    def __init__(self):
        self.model: tf.keras.Model = None
        self.bridge_layers: list[layers.Layer] = None
        