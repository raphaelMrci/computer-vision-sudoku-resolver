from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter OCR manifest entries to keep higher quality samples.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON file.")
    parser.add_argument("--out", required=True, help="Output filtered manifest JSON file.")
    parser.add_argument(
        "--allowed-status",
        default="ok,ok_relaxed,unsolved",
        help="Comma-separated allowed status values. Empty keeps all.",
    )
    parser.add_argument("--min-clues", type=int, default=17, help="Minimum clues to keep entry.")
    parser.add_argument("--min-conf", type=float, default=0.68, help="Minimum mean confidence to keep entry.")
    parser.add_argument("--max-uncertain", type=int, default=12, help="Maximum uncertain clues.")
    parser.add_argument("--require-grid", action="store_true", help="Keep only entries with a valid 9x9 grid.")
    return parser.parse_args()


def _is_valid_grid(raw: Any) -> bool:
    if not isinstance(raw, list) or len(raw) != 9:
        return False
    for row in raw:
        if not isinstance(row, list) or len(row) != 9:
            return False
        if not all(isinstance(v, int) and 0 <= v <= 9 for v in row):
            return False
    return True


def _read_manifest(path: Path) -> tuple[List[Dict[str, Any]], bool]:
    payload = json.loads(path.read_text())
    wrapped = False
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        payload = payload["samples"]
        wrapped = True
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a JSON list or an object with a 'samples' list.")
    return [entry for entry in payload if isinstance(entry, dict)], wrapped


def main() -> int:
    args = parse_args()
    input_path = Path(args.manifest)
    output_path = Path(args.out)
    entries, wrapped = _read_manifest(input_path)
    allowed = {s.strip() for s in args.allowed_status.split(",") if s.strip()}

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for entry in entries:
        status = str(entry.get("status", ""))
        clues = int(entry.get("num_clues_detected", 0))
        conf = float(entry.get("ocr_confidence_mean", 0.0))
        uncertain = int(entry.get("num_uncertain_clues", 999))
        grid = entry.get("grid")

        if allowed and status not in allowed:
            dropped += 1
            continue
        if clues < args.min_clues:
            dropped += 1
            continue
        if conf < args.min_conf:
            dropped += 1
            continue
        if uncertain > args.max_uncertain:
            dropped += 1
            continue
        if args.require_grid and not _is_valid_grid(grid):
            dropped += 1
            continue
        kept.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload: Any = {"samples": kept} if wrapped else kept
    output_path.write_text(json.dumps(output_payload, indent=2))
    print(
        f"Filtered manifest: kept={len(kept)} dropped={dropped} "
        f"(from total={len(entries)}). Output: {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
