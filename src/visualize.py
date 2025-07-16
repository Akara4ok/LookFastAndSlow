import tensorflow as tf
import matplotlib.pyplot as plt

def visulize(image: tf.Tensor, detection_result: dict, label_map: list[str]):
    for imgInd in range(len(detection_result['boxes'].numpy())):
        boxes = detection_result['boxes'].numpy()[imgInd]
        scores = detection_result['scores'].numpy()[imgInd]
        classes = detection_result['classes'].numpy()[imgInd]

        # Вивід
        plt.figure(figsize=(6, 6))
        plt.imshow(image.numpy())
        ax = plt.gca()

        for i in range(detection_result['num_detections'].numpy()[imgInd]):
            # if scores[i] < 0.3:
            #     continue
            box = boxes[i]
            cls = classes[i]
            label = label_map[cls]
            if(label == "background"):
                continue
            score = scores[i]
            ymin, xmin, ymax, xmax = box
            h = ymax - ymin
            w = xmax - xmin
            rect = plt.Rectangle((xmin * 300, ymin * 300), w * 300, h * 300,
                                fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin * 300, ymin * 300 - 5, f"{label} {score:.2f}", color='red', fontsize=8)

        plt.axis(False)
        plt.show()
