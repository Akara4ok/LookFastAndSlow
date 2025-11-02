import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from ObjectDetector.Yolo.Models.test_model import CustomDetect 
import ultralytics.nn.tasks as tasks

tasks.CustomDetect = CustomDetect

model = YOLO("src/ObjectDetector/Yolo/Models/test_model.yaml")
model.load("Model/yolo11x.pt") 

model.train(
    data="Data/YoloVoc/data.yaml",
    epochs=10,
    imgsz=640,
    batch=2
)