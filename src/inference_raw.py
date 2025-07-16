from ConfigUtils.config import Config
from pathlib import Path
import logging
from ObjectDetector.object_detector import ObjectDetector
import tensorflow as tf
from visualize import visulize
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()

labels = [ "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
objectDetector = ObjectDetector(labels, config)
objectDetector.load_weights(config['model']['path'])

image = tf.io.read_file("Data/voc_test/not_voc.jpg")
image = tf.image.decode_jpeg(image, channels=3)
image = tf.cast(image, tf.float32)
image = tf.image.resize(image, (300, 300))

result = objectDetector.predict(preprocess_input(image) )
visulize(image / 255, result, labels)
