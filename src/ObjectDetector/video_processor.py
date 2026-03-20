import time
from typing import Optional
import torch
import cv2

class VideoProcessor:
    def __init__(
        self,
        model,
        device: Optional[str] = None
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
    
    def _put_title(self, frame, text: str, x: int = 10):  # NEW
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, min(w, h) / 1000.0)
        thickness = max(1, int(min(w, h) / 700))

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        y = 10 + th + 6

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, 10), (x + tw + 12, y + 6), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(
            frame,
            text,
            (x + 6, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return frame
    
    def _process_frame_with_model(  # NEW
        self,
        frame,
        model,
        fps_ema: Optional[float] = None,
        alpha: float = 0.1,
        title: Optional[str] = None,
        step_mode: bool = False,
    ):


        processed = frame.copy()
        t0 = time.perf_counter()
        results = model.predict(processed)
        processed = self._draw_boxes(processed, results)
        dt = max(1e-6, time.perf_counter() - t0)

        fps_inst = 1.0 / dt

        if(fps_ema is not None):
            fps_ema = fps_inst if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * fps_inst

        if title is not None:
            processed = self._put_title(processed, title)

        if(not step_mode and fps_ema is not None):
            processed = self._put_fps(processed, fps_ema)

        return processed, fps_ema

    def _make_split_view(  # NEW
        self,
        left_frame,
        right_frame,
    ):
        if left_frame.shape[:2] != right_frame.shape[:2]:
            h, w = left_frame.shape[:2]
            right_frame = cv2.resize(right_frame, (w, h))

        combined = cv2.hconcat([left_frame, right_frame])

        h, w = combined.shape[:2]
        mid_x = w // 2
        cv2.line(combined, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)

        return combined

    def _read_frame_at_index(self, cap, frame_idx: int):  # NEW
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
        step_mode: bool = False,
        output_fps: Optional[float] = None,
        compare_model=None,
        left_title: str = "Main model",
        right_title: str = "Compare model",
    ):
        split_mode = compare_model is not None
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if not fps_in or fps_in <= 0:
            fps_in = 30.0

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path is not None:
            fourcc = (
                cv2.VideoWriter_fourcc(*"mp4v")
                if str(output_path).lower().endswith(".mp4")
                else cv2.VideoWriter_fourcc(*"XVID")
            )

            fps_out = output_fps if output_fps is not None else fps_in

            if split_mode:
                writer_size = (frame_w * 2, frame_h)
            else:
                writer_size = (frame_w, frame_h)

            writer = cv2.VideoWriter(output_path, fourcc, fps_out, writer_size)

        fps_ema = 0.0
        shown_frames = 0
        target_frame_time = None if output_fps is None else 1.0 / output_fps

        try:
            while True:
                loop_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                if split_mode:
                    left_processed, _ = self._process_frame_with_model(
                        frame,
                        model=self.model,
                        title=left_title,
                        fps_ema=None,
                    )

                    right_processed, _ = self._process_frame_with_model(
                        frame,
                        model=compare_model,
                        title=right_title,
                        fps_ema=None,
                    )
                    output_frame = self._make_split_view(left_processed, right_processed)
                else:
                    output_frame, new_fps_left = self._process_frame_with_model(
                        frame,
                        model=self.model,
                        fps_ema=fps_ema,
                        step_mode=step_mode
                    )
                    fps_ema = new_fps_left

                if writer is not None:
                    writer.write(output_frame)

                if display:
                    window_name = "Video Detection Split" if split_mode else "Video Detection"
                    cv2.imshow(window_name, output_frame)

                    if step_mode:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        wait_ms = 1
                        if target_frame_time is not None:
                            elapsed = time.perf_counter() - loop_start
                            remaining = max(0.0, target_frame_time - elapsed)
                            wait_ms = max(1, int(remaining * 1000))

                        key = cv2.waitKey(wait_ms) & 0xFF

                    if(key == ord(' ')):
                        step_mode = not step_mode

                    if(key == 81):
                        shown_frames -= 1
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, shown_frames - 2))
                        continue

                    if key == 27:
                        break

                shown_frames += 1
                if max_frames is not None and shown_frames >= max_frames:
                    break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()