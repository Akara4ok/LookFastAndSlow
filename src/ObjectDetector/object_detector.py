import logging
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from ObjectDetector.Models.lite_mobilenet_backbone import LiteMobileNetBackbone
from ObjectDetector.Models.ssd_head import SSDHead
from ObjectDetector.anchors import Anchors
from ObjectDetector.loss import SSDLoss
from ObjectDetector.postprocessing import PostProcessing

class ObjectDetector():
    def __init__(self, labels: list[str], config: dict) -> None:
        self.config = config
        self.labels = labels
        self.model = self.build_model()
        
        anchor_config = self.config['anchors']
        self.anchors = Anchors(anchor_config['feature_map'], anchor_config['aspects'], anchor_config['variances'])
        
        self.log_model_valid()
        
        self.postprocessing = PostProcessing(self.anchors.anchors, 
                                             anchor_config['variances'], 
                                             anchor_config['confidence'], 
                                             anchor_config['iou_threshold'],
                                             anchor_config['top_k_classes'])
        
    def log_model_valid(self) -> None:
        if(self.model.output[0].shape[1] != len(self.anchors.anchors)):
            logging.warning(f'Model anchors {self.model.output[0].shape[1]} != generated anchors {len(self.anchors.anchors)}')
        
        if(self.model.output[0].shape[2] != len(self.labels)):
            logging.warning(f'Model labels {self.model.output[0].shape[2]} != set labels {len(self.labels)}')
        
    def build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self.config['model']['img_size'])
        backbone = LiteMobileNetBackbone(inputs)
        head = SSDHead(backbone, len(self.labels), self.config['anchors']['aspects'])
        return tf.keras.Model(inputs=inputs, outputs=[head.cls_final, head.loc_final])
    
    def encode_dataset(self, ds: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        def encode_wrapper_tf(image, label_dict):
            boxes = label_dict["boxes"]
            labels = label_dict["labels"]

            loc_targets, cls_targets = self.anchors.match_anchors_to_gt_tf(
                boxes, labels, len(self.labels), self.config['anchors']['iou_threshold']
            )

            return image, (cls_targets, loc_targets)
        
        ds = ds.map(encode_wrapper_tf, num_parallel_calls=tf.data.AUTOTUNE)
        
        img_size = self.config['model']['img_size']
        anchors_count = len(self.anchors.anchors)
        num_labels = len(self.labels)
        
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=(img_size, ([anchors_count, num_labels], [anchors_count, 4])),
            drop_remainder=True
        )
        return ds.prefetch(tf.data.AUTOTUNE)
    
    def train_test_split(self, ds: tf.data.Dataset, test_percent: float) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        dataset_size = sum(1 for _ in ds)
        
        test_size = int(test_percent * dataset_size)

        test_dataset = ds.take(test_size)
        train_dataset = ds.skip(test_size)
        
        return train_dataset, test_dataset
    
    def train(self, ds: tf.data.Dataset) -> None:
        ds = self.encode_dataset(ds, self.config['train']['batch_size'])
        
        custom_loss = SSDLoss()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']['initial_lr']),
            loss=[custom_loss.cls_loss, custom_loss.loc_loss]
        )
        
        train_ds, val_ds = self.train_test_split(ds, self.config['data']['test_percent'])
        fullModelSave = ModelCheckpoint(filepath=self.config['model']['path'],
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto')
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.config['train']['tensorboard_path'], histogram_freq=1)
        
        callbacks_list = [tensorboard_callback, fullModelSave]
        epochs = self.config['train']['epochs']
        self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks_list, verbose=1)

    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)
        
    def predict(self, x: tf.Tensor) -> dict:
        while (len(x.shape) < 4):
            x = tf.expand_dims(x, 0)
        cls_final, loc_final = self.model.predict(x)
        return self.postprocessing.ssd_postprocess(cls_final, loc_final)