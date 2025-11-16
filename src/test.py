import time
import numpy as np
import torch
from ultralytics import YOLO

def measure(model, input_data, runs=50):
    # Warmup
    for _ in range(5):
        _ = model(input_data)
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(input_data)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    return avg


def main():
    device = "cuda"

    # Load YOLO model
    model = YOLO("Model/Yolo/yolo11x_voc.pt")
    model.to(device)
    model.eval()

    # Create inputs
    np_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    tensor_img = torch.randint(0, 255, (3, 640, 640), dtype=torch.uint8).to(device)
    tensor_img = tensor_img.float() / 255.0

    # Wrap tensor to batch dimension (YOLO expects B,C,H,W)
    tensor_img = tensor_img.unsqueeze(0)

    # Measure numpy input (YOLO accepts numpy HWC BGR)
    avg_numpy = measure(model, np_img)

    # Measure tensor input
    # avg_tensor = measure(model, tensor_img)

    print(f"Avg predict time (numpy input):  {avg_numpy * 1000:.3f} ms")
    # print(f"Avg predict time (tensor input): {avg_tensor * 1000:.3f} ms")


if __name__ == "__main__":
    main()
