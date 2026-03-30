from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model for Sudoku grid/cell detection.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--project", default="runs/train", help="Ultralytics project directory.")
    parser.add_argument("--name", default="sudoku-grid-detector", help="Run name.")
    parser.add_argument("--device", default="cpu", help="Device, e.g. cpu, 0, 0,1.")
    parser.add_argument("--export", action="store_true", help="Export best model to ONNX at the end.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )
    model.val()
    if args.export:
        model.export(format="onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
