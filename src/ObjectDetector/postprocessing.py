import tensorflow as tf

class PostProcessing():
    def __init__(self, anchors, variances, confidence=0.5, iou_thresh=0.5, top_k=1):
        self.anchors = anchors
        self.variances = variances
        self.confidence = confidence
        self.iou_threshold = iou_thresh
        self.top_k = top_k

    def decode_boxes(self, pred_loc: tf.Tensor) -> tf.Tensor:
        pred_loc *= self.variances
        ty, tx, th, tw = tf.unstack(pred_loc, axis=-1)

        ymin, xmin, ymax, xmax = tf.unstack(self.anchors, axis=-1)
        h = ymax - ymin
        w = xmax - xmin
        cy = ymin + 0.5 * h
        cx = xmin + 0.5 * w

        pred_cy = ty * h + cy
        pred_cx = tx * w + cx
        pred_h = tf.exp(th) * h
        pred_w = tf.exp(tw) * w

        ymin = pred_cy - pred_h / 2.0
        xmin = pred_cx - pred_w / 2.0
        ymax = pred_cy + pred_h / 2.0
        xmax = pred_cx + pred_w / 2.0

        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def ssd_postprocess(self, cls_logits: tf.Tensor, pred_loc: tf.Tensor) -> dict:
        probs = tf.nn.softmax(cls_logits[0], axis=-1)
        boxes = self.decode_boxes(pred_loc[0])

        boxes_expanded = tf.expand_dims(boxes, axis=1)
        boxes_expanded = tf.expand_dims(boxes_expanded, axis=0)

        scores = tf.expand_dims(probs, axis=0)  # (1, N, C)

        selected = tf.image.combined_non_max_suppression(
            boxes=boxes_expanded,
            scores=scores,
            max_output_size_per_class=self.top_k,
            max_total_size=self.top_k,
            iou_threshold=self.iou_threshold,
            score_threshold=self.confidence,
            clip_boxes=False
        )

        return {
            "boxes": selected.nmsed_boxes,
            "scores": selected.nmsed_scores,
            "classes": tf.cast(selected.nmsed_classes, tf.int32),
            "num_detections": selected.valid_detections
        }
