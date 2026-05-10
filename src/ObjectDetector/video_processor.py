import time
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import torch
import cv2

from ObjectDetector.map import MeanAveragePrecision

class VideoProcessor:
    def __init__(
        self,
        model,
        device: Optional[str] = None
    ):
        self.model = model
        
        self.device = device
        self.labels = self.model.labels

    def _new_runtime_store(self):
        return {
            "predictions": {},   # frame_idx -> {"boxes", "scores", "classes"}
            "latencies_ms": [],
        }

    def _empty_pred(self):
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int64),
        }
    
    def _to_absolute_prediction(self, frame, results):
        if results is None:
            return self._empty_pred()

        h, w = frame.shape[:2]

        boxes = np.asarray(
            results.get("boxes", np.empty((0, 4), dtype=np.float32)),
            dtype=np.float32
        ).copy()
        scores = np.asarray(
            results.get("scores", np.empty((0,), dtype=np.float32)),
            dtype=np.float32
        ).copy()
        classes = np.asarray(
            results.get("classes", np.empty((0,), dtype=np.int64)),
            dtype=np.int64
        ).copy()

        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            boxes = boxes.reshape(-1, 4)
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h

            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
        }
    
    def _store_prediction(self, runtime_store, frame_idx, frame, results, latency_ms):
        runtime_store["predictions"][frame_idx] = self._to_absolute_prediction(frame, results)
        runtime_store["latencies_ms"].append(float(latency_ms))

    def _compute_metrics_snapshot(self, runtime_store, gt_dataset, up_to_frame_idx: int):
        if gt_dataset is None:
            return None

        gt_frame_nums = [
            frame_idx
            for frame_idx in gt_dataset.get_available_frame_nums()
            if frame_idx <= up_to_frame_idx
        ]
        gt_frame_nums = sorted(gt_frame_nums)

        if not gt_frame_nums:
            return None

        preds = []
        targets = []
        gt_counter = Counter()

        for frame_idx in gt_frame_nums:
            pred_np = runtime_store["predictions"].get(frame_idx, self._empty_pred())
            gt_boxes, gt_labels = gt_dataset.get_annotation_by_frame_num(frame_idx)

            gt_boxes = gt_boxes.astype(np.float32, copy=False)
            gt_labels = gt_labels.astype(np.int64, copy=False)
            gt_counter.update(torch.from_numpy(pred_np["classes"]).tolist())

            preds.append({
                "boxes": torch.from_numpy(pred_np["boxes"]),
                "scores": torch.from_numpy(pred_np["scores"]),
                "classes": torch.from_numpy(pred_np["classes"]),
            })
            targets.append({
                "boxes": torch.from_numpy(gt_boxes),
                "labels": torch.from_numpy(gt_labels),
            })

        metric = MeanAveragePrecision(num_classes=len(self.labels))
        metric.update(preds, targets)
        result = metric.compute()

        result["_frames_evaluated"] = len(gt_frame_nums)
        result["_avg_latency_ms"] = (
            float(np.mean(runtime_store["latencies_ms"]))
            if runtime_store["latencies_ms"]
            else None
        )
        result["_top5"] = gt_counter.most_common(5)
        return result

    def _format_stats_lines(self, metrics, include_latency: bool):
        if metrics is None:
            return [
                "GT dataset is not provided.",
                "mAP / precision / recall are unavailable.",
            ]

        lines = [
            f"Frames: {metrics['_frames_evaluated']}",
            f"mAP: {metrics['weighted_mAP']:.4f}",
            f"Precision / Recall: {metrics['precision']:.4f} / {metrics['recall']:.4f}",
        ]

        if include_latency and metrics.get("_avg_latency_ms") is not None:
            lines.append(f"Avg latency: {metrics['_avg_latency_ms']:.2f} ms")

        lines.append("")
        lines.append("Top-5 classes:")

        top5 = metrics.get("_top5", [])
        if not top5:
            lines.append("No GT objects on evaluated frames.")
            return lines

        for rank, (cls_id, count) in enumerate(top5, 1):
            cls_name = (
                self.labels[int(cls_id)]
                if int(cls_id) < len(self.labels)
                else f"class_{cls_id}"
            )
            ap = metrics.get(f"AP_{cls_id}", 0.0)
            precision = metrics.get(f"precision_{cls_id}", 0.0)
            recall = metrics.get(f"recall_{cls_id}", 0.0)
            fp = metrics.get(f"fp_{cls_id}", 0.0)
            gt = metrics.get(f"gt_count_{cls_id}", 0.0)

            lines.append(f"{rank}. {cls_name} ({count})")
            if(gt > 0):
                lines.append(f"   AP: {ap:.4f} | P/R: {precision:.4f} / {recall:.4f}")
            else:
                lines.append(f"   No GT instances, FP count: {fp}")

        return lines
    
    def _render_stats_panel(self, width: int, height: int, title: str, lines):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.50, min(width, height) / 1800.0)
        thickness = max(1, int(min(width, height) / 900))

        x = 24
        y = 40

        cv2.putText(
            canvas, title, (x, y), font, font_scale + 0.15,
            (255, 255, 255), thickness + 1, cv2.LINE_AA
        )
        y += 16
        cv2.line(canvas, (x, y), (width - x, y), (255, 255, 255), 1)
        y += 28

        line_step = max(24, int(28 * font_scale + 12))
        for line in lines:
            if not line:
                y += line_step // 2
                continue

            cv2.putText(
                canvas, line, (x, y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA
            )
            y += line_step

            if y > height - 20:
                break

        return canvas

    def _build_stats_frame(
        self,
        frame_w: int,
        frame_h: int,
        main_metrics,
        compare_metrics=None,
        left_title: str = "Main model",
        right_title: str = "Compare model",
    ):
        if compare_metrics is None:
            return self._render_stats_panel(
                frame_w,
                frame_h,
                f"{left_title} stats",
                self._format_stats_lines(main_metrics, include_latency=True),
            )

        left_panel = self._render_stats_panel(
            frame_w,
            frame_h,
            f"{left_title} stats",
            self._format_stats_lines(main_metrics, include_latency=False),
        )
        right_panel = self._render_stats_panel(
            frame_w,
            frame_h,
            f"{right_title} stats",
            self._format_stats_lines(compare_metrics, include_latency=False),
        )
        return self._make_split_view(left_panel, right_panel)
    
    def _resize_with_aspect(
        self,
        frame,
        target_size: Tuple[int, int],
        pad_color=(0, 0, 0),
    ):
        target_w, target_h = target_size
        src_h, src_w = frame.shape[:2]

        if target_w <= 0 or target_h <= 0:
            return frame

        scale = min(target_w / src_w, target_h / src_h)

        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    def _prepare_frame_for_display(
        self,
        frame,
        window_size: Optional[Tuple[int, int]] = None,
    ):
        if window_size is None:
            return frame
        return self._resize_with_aspect(frame, window_size)

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
    
    def _process_frame_with_model(
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
        dt = max(1e-6, time.perf_counter() - t0)

        processed = self._draw_boxes(processed, results)

        fps_inst = 1.0 / dt
        if fps_ema is not None:
            fps_ema = fps_inst if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * fps_inst

        if title is not None:
            processed = self._put_title(processed, title)

        if not step_mode and fps_ema is not None:
            processed = self._put_fps(processed, fps_ema)

        return processed, fps_ema, results, dt * 1000.0

    def _make_split_view(
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
        gt_dataset=None,
        append_final_stats: bool = True,
        window_size: Optional[Tuple[int, int]] = None
    ):
        split_mode = compare_model is not None
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if not fps_in or fps_in <= 0:
            fps_in = 30.0
        
        fps_out = output_fps if output_fps is not None else fps_in

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path is not None:
            fourcc = (
                cv2.VideoWriter_fourcc(*"mp4v")
                if str(output_path).lower().endswith(".mp4")
                else cv2.VideoWriter_fourcc(*"XVID")
            )

            writer_size = (frame_w * 2, frame_h) if split_mode else (frame_w, frame_h)
            writer = cv2.VideoWriter(output_path, fourcc, fps_out, writer_size)

        fps_ema = 0.0
        shown_frames = 0
        target_frame_time = None if output_fps is None else 1.0 / output_fps

        main_runtime = self._new_runtime_store()
        compare_runtime = self._new_runtime_store() if split_mode else None

        last_frame_idx = -1
        window_name = "Video Detection Split" if split_mode else "Video Detection"

        try:
            while True:
                loop_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                last_frame_idx = max(last_frame_idx, current_frame_idx)

                if split_mode:
                    left_processed, _, left_results, left_latency_ms = self._process_frame_with_model(
                        frame,
                        model=self.model,
                        title=left_title,
                        fps_ema=None,
                    )
                    self._store_prediction(
                        main_runtime,
                        current_frame_idx,
                        frame,
                        left_results,
                        left_latency_ms
                    )

                    right_processed, _, right_results, right_latency_ms = self._process_frame_with_model(
                        frame,
                        model=compare_model,
                        title=right_title,
                        fps_ema=None,
                    )
                    self._store_prediction(
                        compare_runtime,
                        current_frame_idx,
                        frame,
                        right_results,
                        right_latency_ms
                    )

                    output_frame = self._make_split_view(left_processed, right_processed)
                else:
                    output_frame, new_fps_left, left_results, left_latency_ms = self._process_frame_with_model(
                        frame,
                        model=self.model,
                        fps_ema=fps_ema,
                        step_mode=step_mode
                    )
                    fps_ema = new_fps_left

                    self._store_prediction(
                        main_runtime,
                        current_frame_idx,
                        frame,
                        left_results,
                        left_latency_ms
                    )

                if writer is not None:
                    writer.write(output_frame)

                if display:
                    display_frame = self._prepare_frame_for_display(output_frame, window_size)
                    cv2.imshow(window_name, display_frame)

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

                    if(key == ord('s')):
                        main_metrics = self._compute_metrics_snapshot(
                            main_runtime,
                            gt_dataset,
                            current_frame_idx
                        )
                        compare_metrics = (
                            self._compute_metrics_snapshot(
                                compare_runtime,
                                gt_dataset,
                                current_frame_idx
                            )
                            if split_mode else None
                        )

                        stats_frame = self._build_stats_frame(
                            frame_w=frame_w,
                            frame_h=frame_h,
                            main_metrics=main_metrics,
                            compare_metrics=compare_metrics,
                            left_title=left_title,
                            right_title=right_title,
                        )

                        display_stats_frame = self._prepare_frame_for_display(stats_frame, window_size)
                        cv2.imshow(window_name, display_stats_frame)

                        after_stats_key = cv2.waitKey(0) & 0xFF
                        if after_stats_key == 27:
                            break

                    if key == 27:
                        break

                shown_frames += 1
                if max_frames is not None and shown_frames >= max_frames:
                    break

            if append_final_stats and last_frame_idx >= 0:
                main_metrics = self._compute_metrics_snapshot(
                    main_runtime,
                    gt_dataset,
                    last_frame_idx
                )
                compare_metrics = (
                    self._compute_metrics_snapshot(
                        compare_runtime,
                        gt_dataset,
                        last_frame_idx
                    )
                    if split_mode else None
                )

                final_stats_frame = self._build_stats_frame(
                    frame_w=frame_w,
                    frame_h=frame_h,
                    main_metrics=main_metrics,
                    compare_metrics=compare_metrics,
                    left_title=left_title,
                    right_title=right_title,
                )

                if display:
                    display_final_stats = self._prepare_frame_for_display(final_stats_frame, window_size)
                    cv2.imshow(window_name, display_final_stats)

                    cv2.waitKey(0)

            return {
                "main": self._compute_metrics_snapshot(
                    main_runtime,
                    gt_dataset,
                    last_frame_idx
                ) if last_frame_idx >= 0 else None,
                "compare": self._compute_metrics_snapshot(
                    compare_runtime,
                    gt_dataset,
                    last_frame_idx
                ) if (split_mode and last_frame_idx >= 0) else None,
            }

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()