from ConfigUtils.config import Config
from pathlib import Path
import logging
from ObjectDetector.object_detector import ObjectDetector
import tensorflow as tf
from visualize import visulize
logging.basicConfig(level=logging.INFO)

config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()

labels = ['None', 'Star']
objectDetector = ObjectDetector(labels, config)
objectDetector.load_weights(config['model']['path'])

image = tf.io.read_file("Data/images/a (41).jpg")
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, (300, 300))
image = image / 255.0

result = objectDetector.predict(image)
visulize(image, result, labels)
