"""Command-line runner for YOLOv11 roundabout vehicle counting.

Example:
    python count_video.py --model weights/best.pt --video sample_inputs/demo.mp4 --x1 0.05 --y1 0.55 --x2 0.95 --y2 0.55
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vehicle_counter import CountConfig, RoundaboutVehicleCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv11 vehicle counting for roundabout video.")
    parser.add_argument("--model", default="weights/best.pt", help="Path to YOLO best.pt")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output files")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    parser.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"], help="Tracker config")
    parser.add_argument("--direction", default="both", choices=["both", "positive_to_negative", "negative_to_positive"], help="Counting direction")
    parser.add_argument("--resize-width", type=int, default=960, help="Resize video width before inference. Use 0 to keep original.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for quick demo. 0 = full video.")
    parser.add_argument("--x1", type=float, default=0.05, help="Normalized line start x")
    parser.add_argument("--y1", type=float, default=0.55, help="Normalized line start y")
    parser.add_argument("--x2", type=float, default=0.95, help="Normalized line end x")
    parser.add_argument("--y2", type=float, default=0.55, help="Normalized line end y")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    video_path = Path(args.video)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Put best.pt in weights/ or pass --model.")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    config = CountConfig(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        tracker=args.tracker,
        direction=args.direction,
        resize_width=args.resize_width,
        max_frames=args.max_frames,
    )

    def progress(value: float, message: str) -> None:
        print(f"[{value * 100:6.2f}%] {message}", flush=True)

    counter = RoundaboutVehicleCounter(model_path)
    result = counter.process_video(
        input_video_path=video_path,
        output_dir=args.output_dir,
        line_norm=(args.x1, args.y1, args.x2, args.y2),
        config=config,
        progress_callback=progress,
    )

    print("\nDone.")
    print(f"Total count: {result.total_count}")
    print(f"Per class: {result.per_class_count}")
    print(f"Video: {result.output_video_path}")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")


if __name__ == "__main__":
    main()
