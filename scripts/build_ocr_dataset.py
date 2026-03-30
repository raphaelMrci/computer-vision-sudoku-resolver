from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from src.detection.interface import crop_cell, detect_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCR CNN dataset from labeled Sudoku screenshots.")
    parser.add_argument("--manifest", required=True, help="JSON file with samples {image, grid}.")
    parser.add_argument("--out-dir", default="data/ocr_cnn", help="Output dataset directory.")
    return parser.parse_args()


def _parse_grid(value: object) -> List[List[int]]:
    if isinstance(value, str):
        s = "".join(ch for ch in value if ch.isdigit())
        if len(s) != 81:
            raise ValueError("Grid string must contain 81 digits.")
        nums = [int(ch) for ch in s]
    elif isinstance(value, list):
        if len(value) != 9 or any(not isinstance(r, list) or len(r) != 9 for r in value):
            raise ValueError("Grid must be a 9x9 list.")
        nums = [int(v) for row in value for v in row]
    else:
        raise ValueError("Grid must be string or 9x9 list.")
    return [nums[i * 9 : (i + 1) * 9] for i in range(9)]


def _prepare_cell(cell: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    py = max(1, int(h * 0.2))
    px = max(1, int(w * 0.2))
    roi = gray[py : h - py, px : w - px]
    if roi.size == 0:
        roi = gray
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
    return roi


def main() -> int:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError("Manifest must contain a non-empty 'samples' list.")

    out_root = Path(args.out_dir)
    for label in range(10):
        (out_root / str(label)).mkdir(parents=True, exist_ok=True)

    count = 0
    for s_idx, sample in enumerate(samples):
        image = np.array(Image.open(sample["image"]).convert("RGB"))
        grid = _parse_grid(sample["grid"])
        boxes = detect_cells(image)
        for box in boxes:
            label = int(grid[box.row][box.col])
            cell = crop_cell(image, box)
            processed = _prepare_cell(cell)
            out_path = out_root / str(label) / f"s{s_idx:03d}_r{box.row}_c{box.col}.png"
            cv2.imwrite(str(out_path), processed)
            count += 1

    print(f"Saved {count} cell crops to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
