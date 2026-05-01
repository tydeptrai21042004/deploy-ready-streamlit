"""YOLOv11 image/video vehicle detection and line-crossing counter."""
from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .geometry import CountingLine, Direction, box_center_xyxy

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - handled in app UI
    YOLO = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None


ProgressCallback = Callable[[float, str], None]

VEHICLE_NAME_KEYWORDS = (
    "car",
    "vehicle",
    "bus",
    "truck",
    "van",
    "taxi",
    "motorcycle",
    "motorbike",
    "bike",
    "bicycle",
    "cycle",
    "scooter",
    "lorry",
    "trailer",
)


@dataclass
class CountConfig:
    """Runtime configuration for video counting."""

    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 960
    tracker: str = "bytetrack.yaml"
    direction: Direction = "both"
    resize_width: int = 960
    max_frames: int = 0
    draw_track_history: bool = True
    allowed_class_ids: Optional[List[int]] = None


@dataclass
class CountResult:
    """Serializable result returned after processing a video."""

    total_count: int
    per_class_count: Dict[str, int]
    counted_track_ids: List[int]
    processed_frames: int
    input_fps: float
    output_video_path: str
    csv_path: str
    json_path: str
    elapsed_seconds: float


@dataclass
class ImageDetection:
    """One image detection row."""

    object_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float


@dataclass
class ImageCountResult:
    """Serializable result returned after processing one image."""

    total_count: int
    per_class_count: Dict[str, int]
    detections: List[Dict[str, object]]
    input_image_path: str
    output_image_path: str
    csv_path: str
    json_path: str
    elapsed_seconds: float


class RoundaboutVehicleCounter:
    """Vehicle counter based on YOLO detection/tracking.

    Image mode:
    - Runs normal YOLO detection.
    - Counts all detected objects after the selected class filter.

    Video mode:
    - Runs YOLO tracking.
    - Counts a tracked object once when its center crosses the configured line.
    """

    def __init__(self, model_path: str | Path):
        if YOLO is None:
            raise RuntimeError(
                "Cannot import ultralytics. Install requirements.txt first. "
                f"Original error: {_YOLO_IMPORT_ERROR}"
            )
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        self.names = self._normalize_names(getattr(self.model, "names", {}))

    @staticmethod
    def _normalize_names(names) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {idx: str(v) for idx, v in enumerate(names)}
        return {0: "car", 1: "cycle", 2: "bus", 3: "truck", 4: "van"}

    @staticmethod
    def _safe_mkdir(path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _video_writer(path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
        """Create a browser-friendly MP4 writer when possible."""
        fourcc_candidates = ["mp4v", "avc1", "XVID"]
        for code in fourcc_candidates:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, frame_size)
            if writer.isOpened():
                return writer
            writer.release()
        raise RuntimeError("Could not create output video writer. Check ffmpeg/OpenCV installation.")

    @staticmethod
    def _resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
        if target_width <= 0:
            return frame
        height, width = frame.shape[:2]
        if width <= target_width:
            return frame
        scale = target_width / float(width)
        new_size = (target_width, int(round(height * scale)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _deterministic_color(class_id: int) -> Tuple[int, int, int]:
        palette = [
            (56, 142, 60),
            (25, 118, 210),
            (245, 124, 0),
            (142, 36, 170),
            (198, 40, 40),
            (0, 121, 107),
            (93, 64, 55),
            (69, 90, 100),
        ]
        return palette[int(class_id) % len(palette)]

    def auto_vehicle_class_ids(self) -> List[int]:
        """Return class IDs whose model names look like vehicle classes."""
        ids: List[int] = []
        for class_id, name in self.names.items():
            normalized = str(name).lower().replace("_", "-").replace(" ", "-")
            if any(keyword in normalized for keyword in VEHICLE_NAME_KEYWORDS):
                ids.append(int(class_id))
        return sorted(set(ids))

    def model_class_table(self) -> List[Dict[str, object]]:
        """Return model class metadata for display in the UI."""
        return [{"class_id": k, "class_name": v} for k, v in sorted(self.names.items())]

    def _draw_detection_overlay(
        self,
        frame: np.ndarray,
        detections: List[ImageDetection],
        per_class_count: Counter,
        title: str = "Image vehicle count",
    ) -> None:
        """Draw image detections and summary panel in-place on a BGR frame."""
        h, w = frame.shape[:2]

        for det in detections:
            color = self._deterministic_color(det.class_id)
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cx, cy = int(det.center_x), int(det.center_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            label = f"#{det.object_id} {det.class_name} {det.confidence:.2f}"
            label_y = max(22, y1 - 8)
            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        panel_w = min(460, max(320, int(w * 0.45)))
        panel_h = 72 + 26 * max(1, len(per_class_count))
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

        total = int(sum(per_class_count.values()))
        cv2.putText(
            frame,
            title,
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Total vehicles: {total}",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y = 98
        if not per_class_count:
            cv2.putText(frame, "No objects detected", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            for cls_name, cnt in sorted(per_class_count.items()):
                cv2.putText(frame, f"{cls_name}: {cnt}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                y += 26

    def _draw_video_overlay(
        self,
        frame: np.ndarray,
        line: CountingLine,
        per_class_count: Counter,
        total_count: int,
        frame_index: int,
    ) -> None:
        h, w = frame.shape[:2]
        p1 = tuple(map(int, line.p1))
        p2 = tuple(map(int, line.p2))
        cv2.line(frame, p1, p2, (0, 255, 255), 3)
        cv2.circle(frame, p1, 6, (0, 255, 255), -1)
        cv2.circle(frame, p2, 6, (0, 255, 255), -1)

        panel_w = min(430, max(300, int(w * 0.38)))
        panel_h = 38 + 26 * (len(per_class_count) + 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(
            frame,
            f"Total counted: {total_count} | Frame: {frame_index}",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y = 72
        for cls_name, cnt in sorted(per_class_count.items()):
            cv2.putText(
                frame,
                f"{cls_name}: {cnt}",
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 26

    def process_image(
        self,
        input_image_path: str | Path,
        output_dir: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 960,
        allowed_class_ids: Optional[List[int]] = None,
    ) -> ImageCountResult:
        """Detect and count vehicles/objects in a single image."""
        output_dir = self._safe_mkdir(output_dir)
        input_image_path = Path(input_image_path)
        if not input_image_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_image_path}")

        start_time = time.time()
        pil_image = Image.open(input_image_path).convert("RGB")
        frame_rgb = np.array(pil_image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = input_image_path.stem.replace(" ", "_")
        output_image_path = output_dir / f"{stem}_detected_{timestamp}.jpg"
        csv_path = output_dir / f"{stem}_detections_{timestamp}.csv"
        json_path = output_dir / f"{stem}_summary_{timestamp}.json"

        detections: List[ImageDetection] = []
        per_class_count: Counter = Counter()

        # If a manual filter is intentionally empty, skip inference and return zero detections.
        if allowed_class_ids is None or len(allowed_class_ids) > 0:
            results = self.model.predict(
                frame_bgr,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                verbose=False,
                classes=allowed_class_ids,
            )
            result = results[0] if isinstance(results, list) else results
            boxes = getattr(result, "boxes", None)

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
                class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy), dtype=float)

                for object_idx, (box, class_id, det_conf) in enumerate(zip(xyxy, class_ids, confs), start=1):
                    class_id = int(class_id)
                    cls_name = self.names.get(class_id, f"class_{class_id}")
                    center = box_center_xyxy(tuple(map(float, box)))
                    det = ImageDetection(
                        object_id=object_idx,
                        class_id=class_id,
                        class_name=cls_name,
                        confidence=round(float(det_conf), 6),
                        x1=round(float(box[0]), 3),
                        y1=round(float(box[1]), 3),
                        x2=round(float(box[2]), 3),
                        y2=round(float(box[3]), 3),
                        center_x=round(float(center[0]), 3),
                        center_y=round(float(center[1]), 3),
                    )
                    detections.append(det)
                    per_class_count[cls_name] += 1

        self._draw_detection_overlay(frame_bgr, detections, per_class_count)
        cv2.imwrite(str(output_image_path), frame_bgr)

        rows = [asdict(det) for det in detections]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "object_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
            ]
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(rows)

        result = ImageCountResult(
            total_count=int(sum(per_class_count.values())),
            per_class_count=dict(sorted(per_class_count.items())),
            detections=rows,
            input_image_path=str(input_image_path),
            output_image_path=str(output_image_path),
            csv_path=str(csv_path),
            json_path=str(json_path),
            elapsed_seconds=round(time.time() - start_time, 3),
        )
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        return result

    def process_video(
        self,
        input_video_path: str | Path,
        output_dir: str | Path,
        line_norm: Tuple[float, float, float, float] = (0.05, 0.55, 0.95, 0.55),
        config: Optional[CountConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> CountResult:
        config = config or CountConfig()
        output_dir = self._safe_mkdir(output_dir)
        input_video_path = Path(input_video_path)
        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_frames = int(config.max_frames or 0)
        total_frames_for_progress = max_frames if max_frames > 0 else total_frames_raw

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            raise RuntimeError("Cannot read the first frame from the input video.")

        first_frame = self._resize_keep_aspect(first_frame, config.resize_width)
        height, width = first_frame.shape[:2]
        line = CountingLine.from_normalized(*line_norm, width=width, height=height)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = input_video_path.stem.replace(" ", "_")
        output_video_path = output_dir / f"{stem}_counted_{timestamp}.mp4"
        csv_path = output_dir / f"{stem}_counts_{timestamp}.csv"
        json_path = output_dir / f"{stem}_summary_{timestamp}.json"
        writer = self._video_writer(output_video_path, fps=fps, frame_size=(width, height))

        # Reset capture after first-frame probing.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        track_last_side: Dict[int, float] = {}
        counted_track_ids: set[int] = set()
        track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        per_class_count: Counter = Counter()
        rows: List[Dict[str, object]] = []

        # Reset possible previous tracker state if the same model instance is reused.
        try:
            self.model.predictor = None
        except Exception:
            pass

        frame_index = 0
        written_frames = 0
        start_time = time.time()

        while True:
            if max_frames > 0 and frame_index >= max_frames:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_index += 1
            frame = self._resize_keep_aspect(frame, config.resize_width)
            results = self.model.track(
                frame,
                persist=True,
                conf=float(config.conf),
                iou=float(config.iou),
                imgsz=int(config.imgsz),
                tracker=str(config.tracker),
                verbose=False,
                classes=config.allowed_class_ids,
            )
            result = results[0] if isinstance(results, list) else results
            boxes = getattr(result, "boxes", None)

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
                class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy), dtype=float)
                ids_tensor = boxes.id
                track_ids = ids_tensor.cpu().numpy().astype(int) if ids_tensor is not None else np.array([-1] * len(xyxy))

                for box, class_id, conf, track_id in zip(xyxy, class_ids, confs, track_ids):
                    if track_id < 0:
                        continue

                    cls_name = self.names.get(int(class_id), f"class_{int(class_id)}")
                    color = self._deterministic_color(int(class_id))
                    x1, y1, x2, y2 = map(int, box)
                    center = box_center_xyxy(tuple(map(float, box)))
                    cx, cy = map(int, center)

                    track_history[int(track_id)].append((cx, cy))
                    current_side = line.side(center)
                    previous_side = track_last_side.get(int(track_id))
                    track_last_side[int(track_id)] = current_side

                    crossed = False
                    if previous_side is not None and int(track_id) not in counted_track_ids:
                        crossed = line.crossed(previous_side, current_side, config.direction)
                        if crossed:
                            counted_track_ids.add(int(track_id))
                            per_class_count[cls_name] += 1
                            rows.append(
                                {
                                    "frame": frame_index,
                                    "track_id": int(track_id),
                                    "class_id": int(class_id),
                                    "class_name": cls_name,
                                    "confidence": float(conf),
                                    "center_x": float(center[0]),
                                    "center_y": float(center[1]),
                                    "count_total_after_event": int(sum(per_class_count.values())),
                                }
                            )

                    is_counted = int(track_id) in counted_track_ids
                    thickness = 3 if is_counted else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"ID {track_id} {cls_name} {conf:.2f}"
                    if is_counted:
                        label += " COUNTED"
                    cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

                    if config.draw_track_history:
                        history = list(track_history[int(track_id)])
                        for idx in range(1, len(history)):
                            cv2.line(frame, history[idx - 1], history[idx], color, 2)

                    if crossed:
                        cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 3)

            self._draw_video_overlay(frame, line, per_class_count, int(sum(per_class_count.values())), frame_index)
            writer.write(frame)
            written_frames += 1

            if progress_callback is not None:
                if total_frames_for_progress > 0:
                    progress = min(frame_index / float(total_frames_for_progress), 1.0)
                else:
                    progress = 0.0
                progress_callback(progress, f"Processing frame {frame_index}/{total_frames_for_progress or '?'}")

        cap.release()
        writer.release()

        total_count = int(sum(per_class_count.values()))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "frame",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "center_x",
                "center_y",
                "count_total_after_event",
            ]
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(rows)

        result = CountResult(
            total_count=total_count,
            per_class_count=dict(sorted(per_class_count.items())),
            counted_track_ids=sorted(counted_track_ids),
            processed_frames=written_frames,
            input_fps=fps,
            output_video_path=str(output_video_path),
            csv_path=str(csv_path),
            json_path=str(json_path),
            elapsed_seconds=round(time.time() - start_time, 3),
        )
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        if progress_callback is not None:
            progress_callback(1.0, "Done")

        return result
