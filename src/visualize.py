import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visulize(image: Image, detection_result: dict, label_map: list[str]):
    boxes = detection_result['boxes']
    scores = detection_result['scores']
    classes = detection_result['classes']

    # Вивід
    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(image))
    ax = plt.gca()
    
    for i in range(boxes.shape[0]):
        if scores[i] < 0.3:
            continue
        box = boxes[i]
        cls = classes[i]
        print(cls)
        label = label_map[cls]
        if(label == "background"):
            continue
        score = scores[i]
        xmin, ymin, xmax, ymax = box
        h = ymax - ymin
        w = xmax - xmin

        img_w = image.shape[1]
        img_h = image.shape[0]
        rect = plt.Rectangle((xmin * img_w, ymin * img_h), w * img_w, h * img_h,
                            fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin * img_w, ymin * img_h - 5, f"{label} {score:.2f}", color='red', fontsize=8)

    plt.axis(False)
    plt.show()
