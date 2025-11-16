import time
from typing import Optional

import cv2

class VideoProcessor:
    def __init__(
        self,
        model,
        device: Optional[str] = None,
    ):
        self.model = model
        
        self.device = device
        self.labels = self.model.labels

    def _draw_boxes(self, frame, results):
        #results: {'boxes': [], 'scores': [], 'classes': []}
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w, h) / 900.0)
        thickness = max(1, int(min(w, h) / 600))

        if results is None:
            return frame
        
        xyxy = results["boxes"]
        confs = results["scores"]
        clss = results["classes"].astype(int)

        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            x1 = x1 * w
            x2 = x2 * w
            y1 = y1 * h
            y2 = y2 * h
            
            label = self.labels[cls]
            color = (0, 255, 0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            y_text = max(int(y1) - 5, th + 5)
            cv2.rectangle(frame, (int(x1), y_text - th - 6), (int(x1) + tw + 6, y_text + 4), color, -1)
            cv2.putText(frame, text, (int(x1) + 3, y_text), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
        return frame

    def _put_fps(self, frame, fps_ema: float):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.7, min(w, h) / 800.0)
        thickness = max(1, int(min(w, h) / 600))
        text = f"FPS: {fps_ema:5.1f}"

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x2 = w - 10
        y2 = 10 + th + 10
        x1 = x2 - tw - 20
        y1 = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, text, (x1 + 10, y2 - 10), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
    ):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") if str(output_path).lower().endswith(".mp4") \
                else cv2.VideoWriter_fourcc(*"XVID")
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

        fps_ema = 0.0
        alpha = 0.1
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.perf_counter()
                results = self.model.predict(frame)
                # frame = self._draw_boxes(frame, results)

                dt = max(1e-6, time.perf_counter() - t0)
                fps_inst = 1.0 / dt
                fps_ema = fps_inst if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * fps_inst
                frame = self._put_fps(frame, fps_ema)

                if writer is not None:
                    writer.write(frame)
                if display:
                    frame = cv2.resize(frame, (1400, 800))
                    cv2.imshow("Video Detection", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                        break

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()