import tensorflow as tf

class SSDLoss():
    def __init__(self):
        self.loc_base = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
        self.cls_base = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE, from_logits=True)
        
    def loc_loss(self, gt_deltas: tf.Tensor, pred_deltas: tf.Tensor) -> tf.Tensor:
        loc_loss_value = self.loc_base(gt_deltas, pred_deltas)

        cond_pos = tf.reduce_any(tf.not_equal(gt_deltas, tf.constant(0.0)), 2)
        mask = tf.cast(cond_pos, tf.float32)
        pos_boxes = tf.reduce_sum(mask, 1)
        
        loc_loss_value = tf.reduce_sum(mask * loc_loss_value, -1)
        pos_boxes = tf.where(tf.equal(pos_boxes, tf.constant(0.0)), tf.constant(1.0), pos_boxes)
        return loc_loss_value / pos_boxes

    def cls_loss(self, gt_labels: tf.Tensor, pred_labels: tf.Tensor) -> tf.Tensor:
        cls_loss_value = self.cls_base(gt_labels, pred_labels)
        
        cond_pos = tf.reduce_any(tf.not_equal(gt_labels[..., 1:], tf.constant(0.0)), 2)
        mask = tf.cast(cond_pos, dtype=tf.float32)
        pos_boxes = tf.reduce_sum(mask, 1)
        neg_pos_boxes = tf.cast(3 * pos_boxes, tf.int32)
        
        masked = cls_loss_value * gt_labels[..., 0]
        loss = tf.argsort(masked, direction="DESCENDING")
        ranks = tf.argsort(loss)
        neg_mask = tf.cast(tf.less(ranks, tf.expand_dims(neg_pos_boxes, 1)), dtype=tf.float32)
        
        final_mask = mask + neg_mask
        cls_loss_value = tf.reduce_sum(final_mask * cls_loss_value, -1)
        pos_boxes = tf.where(tf.equal(pos_boxes, tf.constant(0.0)), tf.constant(1.0), pos_boxes)
        return cls_loss_value / pos_boxes