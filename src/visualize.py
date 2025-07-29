import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visulize(image: Image, detection_result: dict, label_map: list[str]):
    boxes = detection_result['boxes'].numpy()
    scores = detection_result['scores'].numpy()
    classes = detection_result['classes'].numpy()

    # Вивід
    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(image))
    ax = plt.gca()
    
    for i in range(detection_result['num_detections'].numpy()[0]):
        # if scores[i] < 0.3:
        #     continue
        box = boxes[i]
        cls = classes[i]
        label = label_map[cls]
        if(label == "background"):
            continue
        score = scores[i]
        xmin, ymin, xmax, ymax = box
        h = ymax - ymin
        w = xmax - xmin
        rect = plt.Rectangle((xmin * 300, ymin * 300), w * 300, h * 300,
                            fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin * 300, ymin * 300 - 5, f"{label} {score:.2f}", color='red', fontsize=8)

    plt.axis(False)
    plt.show()
