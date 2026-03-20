import sys
import subprocess

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <model> <type>")
        sys.exit(1)

    model = sys.argv[1]
    run_type = sys.argv[2]

    if(run_type == "video"):
        if(model == "yolo11n"):
            subprocess.run(["python3", "src/yolo_custom_training/test_video.py"])
        elif(model == "yolo11x"):
            subprocess.run(["python3", "src/yolo_custom_training/test_video large.py"])
        elif(model == "YoloFastAndSlow"):
            subprocess.run(["python3", "src/yolo_custom_video_training/test_video 2.py"])
        elif(model == "ssd"):
            subprocess.run(["python3", "src/ssd_training/video_testing/SSDLite.py"])
    if(run_type == "test"):
        if(model == "yolo11n"):
            subprocess.run(["python3", "src/yolo_custom_training/test seq.py"])
        elif(model == "yolo11x"):
            subprocess.run(["python3", "src/yolo_custom_training/test seq large.py"])
        elif(model == "YoloFastAndSlow"):
            subprocess.run(["python3", "src/yolo_custom_training/test.py"])
        elif(model == "ssd"):
            subprocess.run(["python3", "src/ssd_training/voc_training/test.py"])
    else:
        print("Unknown type. Use 'video' or 'test'")

if __name__ == "__main__":
    main()