from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OCR approaches on labeled Sudoku screenshots.")
    parser.add_argument("--manifest", required=True, help="Path to benchmark manifest JSON.")
    parser.add_argument("--out", default="artifacts/ocr_benchmark.json", help="Output metrics JSON path.")
    parser.add_argument(
        "--modes",
        default="basic,advanced,hybrid",
        help="Comma-separated OCR modes (e.g. basic,advanced,hybrid,cnn,hybrid_cnn).",
    )
    return parser.parse_args()


def _parse_grid(value: object) -> List[List[int]]:
    if isinstance(value, str):
        s = "".join(ch for ch in value if ch.isdigit())
        if len(s) != 81:
            raise ValueError("Grid string must contain exactly 81 digits.")
        nums = [int(ch) for ch in s]
    elif isinstance(value, list):
        if len(value) != 9 or any(not isinstance(row, list) or len(row) != 9 for row in value):
            raise ValueError("Grid list must be 9x9.")
        nums = [int(v) for row in value for v in row]
    else:
        raise ValueError("Grid must be either an 81-digit string or a 9x9 list.")
    return [nums[i * 9 : (i + 1) * 9] for i in range(9)]


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _evaluate_sample(pred: List[List[int]], gt: List[List[int]]) -> Dict[str, float]:
    correct_all = 0
    total_all = 81
    correct_filled = 0
    total_filled = 0
    pred_filled = 0
    true_positive_filled = 0

    for r in range(9):
        for c in range(9):
            p = pred[r][c]
            g = gt[r][c]
            if p == g:
                correct_all += 1
            if g != 0:
                total_filled += 1
                if p == g:
                    correct_filled += 1
            if p != 0:
                pred_filled += 1
                if p == g and g != 0:
                    true_positive_filled += 1

    return {
        "cell_accuracy_all": _safe_div(correct_all, total_all),
        "digit_accuracy_on_filled_gt": _safe_div(correct_filled, total_filled),
        "filled_precision": _safe_div(true_positive_filled, pred_filled),
        "filled_recall": _safe_div(true_positive_filled, total_filled),
    }


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text())
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError("Manifest must contain a non-empty 'samples' list.")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: Dict[str, Dict[str, object]] = {}

    for mode in modes:
        per_sample = []
        latencies_ms: List[float] = []
        acc_all: List[float] = []
        acc_filled: List[float] = []
        filled_precision: List[float] = []
        filled_recall: List[float] = []

        for sample in samples:
            image_path = Path(sample["image"])
            gt_grid = _parse_grid(sample["grid"])
            image = np.array(Image.open(image_path).convert("RGB"))

            t0 = time.perf_counter()
            pred = run_pipeline(image, ocr_mode=mode)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)

            metrics = _evaluate_sample(pred["initial_grid"], gt_grid)
            acc_all.append(metrics["cell_accuracy_all"])
            acc_filled.append(metrics["digit_accuracy_on_filled_gt"])
            filled_precision.append(metrics["filled_precision"])
            filled_recall.append(metrics["filled_recall"])

            per_sample.append(
                {
                    "image": str(image_path),
                    "latency_ms": latency_ms,
                    "num_clues_detected": pred["num_clues_detected"],
                    **metrics,
                }
            )

        results[mode] = {
            "summary": {
                "num_samples": len(samples),
                "latency_ms_mean": _mean(latencies_ms),
                "cell_accuracy_all_mean": _mean(acc_all),
                "digit_accuracy_on_filled_gt_mean": _mean(acc_filled),
                "filled_precision_mean": _mean(filled_precision),
                "filled_recall_mean": _mean(filled_recall),
            },
            "per_sample": per_sample,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved benchmark report to: {out_path}")
    print(json.dumps({mode: results[mode]['summary'] for mode in modes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
