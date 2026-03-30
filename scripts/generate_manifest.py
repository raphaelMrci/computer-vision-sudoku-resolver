from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate manifest JSON from screenshot directory. "
            "Optionally prefill grid with OCR predictions."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw_screenshots",
        help="Directory containing screenshots.",
    )
    parser.add_argument(
        "--out",
        default="data/manifest_autogen.json",
        help="Output manifest path.",
    )
    parser.add_argument(
        "--extensions",
        default=".png,.jpg,.jpeg,.webp",
        help="Comma-separated list of file extensions.",
    )
    parser.add_argument(
        "--prefill-ocr",
        action="store_true",
        help="Fill grid with OCR prediction to create a train-ready manifest quickly.",
    )
    parser.add_argument(
        "--ocr-mode",
        default="hybrid",
        choices=["advanced", "basic", "hybrid", "cnn", "hybrid_cnn", "tesseract", "hybrid_tesseract"],
        help="OCR mode used when --prefill-ocr is enabled.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store image paths relative to manifest location.",
    )
    return parser.parse_args()


def _flatten_grid(grid: List[List[int]]) -> str:
    return "".join(str(int(v)) for row in grid for v in row)


def _list_images(input_dir: Path, extensions: List[str]) -> List[Path]:
    ext_set = {e.lower().strip() for e in extensions}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in ext_set]
    return sorted(files, key=lambda p: p.name)


def main() -> int:
    try:
        from src.pipeline import run_pipeline
    except ModuleNotFoundError as exc:  # pragma: no cover
        missing = getattr(exc, "name", "dependency")
        print(f"Missing dependency: {missing}")
        print("Install project dependencies first: `make install`")
        return 1

    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ext_list = [e if e.startswith(".") else f".{e}" for e in args.extensions.split(",") if e.strip()]
    images = _list_images(input_dir, ext_list)
    if not images:
        raise ValueError(f"No images found in {input_dir} with extensions {ext_list}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for idx, image_path in enumerate(images):
        if args.relative_paths:
            image_ref = str(image_path.relative_to(out_path.parent))
        else:
            image_ref = str(image_path.resolve())

        sample = {"image": image_ref}
        if args.prefill_ocr:
            image = np.array(Image.open(image_path).convert("RGB"))
            result = run_pipeline(image, ocr_mode=args.ocr_mode)
            sample["grid"] = _flatten_grid(result["initial_grid"])
            sample["ocr_mode"] = args.ocr_mode
            sample["num_clues_detected"] = int(result.get("num_clues_detected", 0))
            sample["ocr_confidence_mean"] = float(result.get("ocr_confidence_mean", 0.0))
            sample["status"] = result.get("status", "unknown")
        else:
            sample["grid"] = ""

        samples.append(sample)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(images):
            print(f"processed {idx + 1}/{len(images)}")

    payload = {"samples": samples}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
