import os
import xml.etree.ElementTree as ET
import tensorflow as tf

class XMLStarDataset():
    def __init__(self, dataset_path: str, img_size: tuple[int]) -> None:
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.data, self.labels = self.build_dataset()

    def parse_voc_xml(self, xml_path: str) -> tuple[str, list, list]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(label)

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / width
            ymin = float(bbox.find('ymin').text) / height
            xmax = float(bbox.find('xmax').text) / width
            ymax = float(bbox.find('ymax').text) / height
            boxes.append([ymin, xmin, ymax, xmax])  # у форматі [ymin, xmin, ymax, xmax] як очікує TensorFlow

        return filename, boxes, labels

    def load_dataset(self, annotation_dir: str, images_dir: str) -> tuple[dict, dict]:
        all_data = []
        label_map = {}
        label_counter = 1  # 0 — фон

        for xml_file in os.listdir(annotation_dir):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(annotation_dir, xml_file)
            filename, boxes, labels = self.parse_voc_xml(xml_path)

            # Конвертація класів у індекси
            label_ids = []
            for label in labels:
                if label not in label_map:
                    label_map[label] = label_counter
                    label_counter += 1
                label_ids.append(label_map[label])

            all_data.append({
                "image_path": os.path.join(images_dir, filename),
                "boxes": boxes,
                "labels": label_ids
            })

        return all_data, label_map

    def load_image_and_labels(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(example['image_path'])
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (self.img_size[0], self.img_size[1]))
        image = image / 255.0

        boxes = tf.convert_to_tensor(example['boxes'], dtype=tf.float32)
        labels = tf.convert_to_tensor(example['labels'], dtype=tf.int32)

        return image, {"boxes": boxes, "labels": labels}

    def build_dataset(self, shuffle: bool = True) -> tuple[tf.data.Dataset, list[str]]:
        dataset_info, label_map = self.load_dataset(self.dataset_path + "/annotations", self.dataset_path + "/images")
        labels = [s for s, i in sorted(label_map.items(), key=lambda x: x[1])]

        ds = tf.data.Dataset.from_generator(
            lambda: iter(dataset_info),
            output_types={"image_path": tf.string, "boxes": tf.float32, "labels": tf.int32}
        )

        if shuffle:
            ds = ds.shuffle(buffer_size=1000)

        ds = ds.map(self.load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
        
        return ds, labels