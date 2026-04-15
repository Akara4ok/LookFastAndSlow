import argparse

import logging
from pathlib import Path

from ConfigUtils.config import Config
from Dataset.image_video_seq_dataset import ImageSeqVideoDataset
from ObjectDetector.SSDLite.image_object_detector import ImageObjectDetector
from ObjectDetector.Yolo.custom_image_object_detector import CustomImageObjectDetector
from ObjectDetector.Yolo.custom_video_object_detector import CustomVideoObjectDetector
from ObjectDetector.Yolo.fast_slow_ssd_video_object_detector import FastSlowSSDVideoObjectDetector
from ObjectDetector.Yolo.general_image_object_detector import GeneralImageObjectDetector
from ObjectDetector.Yolo.yolo_image_seq_tester import YoloImageSeqTester
from ObjectDetector.video_processor import VideoProcessor
from Dataset.voc_dataset import VOCDataset
from ObjectDetector.SSDLite.Anchors.mobilenet_anchors import specs

def parse_args():
    parser = argparse.ArgumentParser(description="My script")

    parser.add_argument("--model", type=str)
    parser.add_argument("--model2", type=str, default=None)
    parser.add_argument("--video", type=str)
    parser.add_argument("--test", action="store_true")

    return parser.parse_args()

def create_model(config: dict, model_name: str, test: bool) -> GeneralImageObjectDetector:
    if model_name == None:
        return None
    
    model_path = f'Model/Final/{model_name}.pt'
    
    if(model_name.lower() in {"yolo11n", "yolo11x"}):
        if(not test):
            objectDetector = CustomImageObjectDetector(config, VOCDataset.VOC_CLASSES)
        else:
            objectDetector = YoloImageSeqTester(config, VOCDataset.VOC_CLASSES)
        objectDetector.load_weights(model_path)
        return objectDetector
    
    if(model_name.lower() == "yolofastandslow"):
        objectDetector = CustomVideoObjectDetector(config, VOCDataset.VOC_CLASSES, True)
        objectDetector.load_weights(model_path, "Model/Final/yolo11n.pt", "Model/Final/yolo11x.pt")
        if(not test):
            objectDetector.set_nms_params(0.45, 0.05)
        return objectDetector
    
    if(model_name.lower() == "ssd"):
        config['model']['img_size'] = 300
        config['anchors']['post_iou_threshold'] = 0.2
        config['anchors']['confidence'] = 0.6
        config['anchors']['top_k_classes'] = 200
        inference = not test
        objectDetector = ImageObjectDetector(['background'] + VOCDataset.VOC_CLASSES, config, specs, None, inference)
        objectDetector.load_weights('Model/Final/model.weights.h5')
        return objectDetector

    if(model_name.lower() == "ssdfastandslow"):
        objectDetector = FastSlowSSDVideoObjectDetector(config, VOCDataset.VOC_CLASSES)
        objectDetector.load_weights(model_path)
        return objectDetector
    
    return None
        
def perform_test(objectDetector: GeneralImageObjectDetector, model_name: str) -> float:
    img_to_test = 96
    voc_ds = VOCDataset("Data/VOCDevKitTest", "2007", "test", use_cache=False)

    if (model_name.lower() != "ssd"):
        voc_ds = ImageSeqVideoDataset(voc_ds)

    return objectDetector.test(voc_ds, img_to_test)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = Config(Path.cwd() / "src/Configs/train.yml").get_dict()
    config['train']['batch_size'] = 1
    config['model']['img_size'] = 640

    args = parse_args()

    print("model:", args.model)
    print("model2:", args.model2)
    print("video:", args.video)
    print("test:", args.test)

    objectDetector1 = create_model(config, args.model, args.test)
    objectDetector2 = create_model(config, args.model2, args.test)

    if(args.video is not None):
        video_path = args.video
        videoProcessor = VideoProcessor(objectDetector1)
        videoProcessor.process_video(f'Data/video/{video_path}.mp4', "Data/output.mp4", True, step_mode=False, output_fps=30, compare_model=objectDetector2, left_title=args.model, right_title=args.model2)
    
    if(args.test):
        print(perform_test(objectDetector1, args.model))
    
