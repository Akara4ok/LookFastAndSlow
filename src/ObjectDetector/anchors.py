import tensorflow as tf

min_scale = 0.2
max_scale = 0.9

class Anchors():
    def __init__(self, feature_map_shapes: list[int], aspect_ratios: list[list[float]], variances: list[float]) -> None:
        self.variances = variances
        
        self.anchors = Anchors.generate_all_anchors(feature_map_shapes, aspect_ratios)
        
    def generate_anchors_for_layer(feature_map_index: int, aspect_ratio: list[float], total_maps: int) -> tf.Tensor:
        def get_scale(map_index: int):
            return min_scale + ((max_scale - min_scale) / (total_maps - 1)) * (map_index - 1)
        current_scale = get_scale(feature_map_index)
        next_scale = get_scale(feature_map_index + 1)
        
        anchors = []
        for aspect_ratio in aspect_ratio:
            height = current_scale / tf.sqrt(aspect_ratio)
            width = current_scale * tf.sqrt(aspect_ratio)
            anchors.append([-height/2, -width/2, height/2, width/2])
            
        height = tf.sqrt(current_scale * next_scale)
        width = tf.sqrt(current_scale * next_scale)
        anchors.append([-height/2, -width/2, height/2, width/2])
        return tf.cast(anchors, dtype=tf.float32)
    
    def generate_all_anchors(feature_map_shapes: list[int], aspect_ratios: list[float]) -> tf.Tensor:
        all_anchors = []

        for i, shape in enumerate(feature_map_shapes):
            base_anchors = Anchors.generate_anchors_for_layer(i+1, aspect_ratios[i], len(feature_map_shapes))
            stride = 1 / shape
            grid = tf.cast(tf.range(0, shape) / shape + stride / 2, dtype=tf.float32)
            x, y = tf.meshgrid(grid, grid)
            x, y = tf.reshape(x, (-1, )), tf.reshape(y, (-1, ))
            xy_map = tf.stack([y, x, y, x], -1)
            anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(xy_map, (-1, 1, 4))
            anchors = tf.reshape(anchors, (-1, 4))
            all_anchors.append(anchors)
        
        anchors = tf.concat(all_anchors, 0)
        return tf.clip_by_value(anchors, 0, 1)

    def yxhw_to_corners_tf(boxes: tf.Tensor) -> tf.Tensor:
        cy, cx, h, w = tf.unstack(boxes, axis=-1)
        ymin = cy - h / 2.0
        xmin = cx - w / 2.0
        ymax = cy + h / 2.0
        xmax = cx + w / 2.0
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def compute_iou_tf(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
        boxes1 = tf.expand_dims(boxes1, 1)
        boxes2 = tf.expand_dims(boxes2, 0)

        ymin = tf.maximum(boxes1[..., 0], boxes2[..., 0])
        xmin = tf.maximum(boxes1[..., 1], boxes2[..., 1])
        ymax = tf.minimum(boxes1[..., 2], boxes2[..., 2])
        xmax = tf.minimum(boxes1[..., 3], boxes2[..., 3])

        inter_area = tf.maximum(ymax - ymin, 0) * tf.maximum(xmax - xmin, 0)

        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        union_area = area1 + area2 - inter_area
        return inter_area / (union_area + 1e-8)

    def compute_delta(boxes: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
        gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
        gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
        gt_cx = gt_boxes[..., 1] + 0.5 * gt_width
        gt_cy = gt_boxes[..., 0] + 0.5 * gt_height
        
        box_width = boxes[..., 3] - boxes[..., 1]
        box_height = boxes[..., 2] - boxes[..., 0]
        box_cx = boxes[..., 1] + 0.5 * box_width
        box_cy = boxes[..., 0] + 0.5 * box_height

        box_width = tf.where(tf.equal(box_width, 0), 1e-3, box_width)
        box_height = tf.where(tf.equal(box_height, 0), 1e-3, box_height)
        
        delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_cx - box_cx), box_width))
        delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_cy - box_cy), box_height))
        delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / box_width))
        delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / box_height))

        return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)

    def match_anchors_to_gt_tf(self, gt_boxes: tf.Tensor, gt_labels: tf.Tensor, num_labels: int, iou_threshold: float = 0.5) -> tf.Tensor:
        iou = Anchors.compute_iou_tf(self.anchors, gt_boxes)
        best_indices = tf.argmax(iou, 1)
        best_iou = tf.reduce_max(iou, 1)

        cond_pos = best_iou > iou_threshold
        
        mboxes = tf.gather(gt_boxes, best_indices)
        mlabels = tf.gather(gt_labels, best_indices)

        mboxes = tf.where(
            tf.expand_dims(cond_pos, -1),
            mboxes,
            tf.zeros_like(mboxes)
        )

        mlabels = tf.where(
            cond_pos,
            mlabels,
            tf.zeros_like(mlabels)
        )

        deltas = Anchors.compute_delta(self.anchors, mboxes) / self.variances
        labels = tf.one_hot(mlabels, num_labels)

        return deltas, labels