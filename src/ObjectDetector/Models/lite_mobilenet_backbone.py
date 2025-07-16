import tensorflow as tf
from tensorflow.keras import layers, models, activations
from ObjectDetector.Models.backbone import Backbone

class LiteMobileNetBackbone(Backbone):
    def __init__(self, inputs: tf.Tensor):
        super().__init__()
        self.model, self.bridge_layers = self.build_model(inputs)

    def inverted_residual_block(self, x, inp, oup, stride, expand_ratio):
        hidden_dim = int(inp * expand_ratio)
        use_res_connect = stride == 1 and inp == oup
    
        out = x
    
        if expand_ratio != 1:
            # pointwise
            out = layers.Conv2D(hidden_dim, 1, padding='same', use_bias=False)(out)
            out = layers.BatchNormalization()(out)
            out = layers.ReLU(max_value=6.0)(out)
    
        # depthwise
        out = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(out)
        out = layers.BatchNormalization()(out)
        out = layers.ReLU(max_value=6.0)(out)
    
        # pointwise-linear
        out = layers.Conv2D(oup, 1, padding='same', use_bias=False)(out)
        out = layers.BatchNormalization()(out)
    
        if use_res_connect:
            out = layers.Add()([x, out])
    
        return out
    
    def build_model(self, inputs: tf.Tensor) -> tuple[tf.keras.Model, list[layers.Layer]]:
        base_model = tf.keras.applications.MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
    
        first_layer = base_model.get_layer("block_13_expand_relu").output
        second_layer = base_model.output
    
        extra1 = self.inverted_residual_block(second_layer, 1280, 512, 2, 0.2)
        extra2 = self.inverted_residual_block(extra1, 512, 256, 2, 0.25)
        extra3 = self.inverted_residual_block(extra2, 256, 256, 2, 0.5)
        extra4 = self.inverted_residual_block(extra3, 256, 64, 2, 0.25)
    
        return base_model, [first_layer, second_layer, extra1, extra2, extra3, extra4]