import cv2
from skimage.metrics import structural_similarity as ssim


def average_ssim_in_video(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Не вдалося відкрити відео: {video_path}")

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Відео порожнє або не вдалося прочитати перший кадр.")

    # Переводимо перший кадр у grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    ssim_values = []
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Якщо розміри раптом різні — підганяємо
        if gray.shape != prev_gray.shape:
            gray = cv2.resize(gray, (prev_gray.shape[1], prev_gray.shape[0]))

        score = ssim(prev_gray, gray)
        ssim_values.append(score)

        prev_gray = gray
        frame_idx += 1
        print(frame_idx, score)

    cap.release()

    if not ssim_values:
        return 0.0

    return min(ssim_values)  # або np.mean(ssim_values) для середнього значення


if __name__ == "__main__":
    video_path = 'Data/video/3.mp4'
    mean_ssim = average_ssim_in_video(video_path)
    print(f"Середній SSIM між сусідніми кадрами: {mean_ssim:.6f}")