from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR benchmark output against reliability/latency targets.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON output.")
    parser.add_argument("--out", default="", help="Optional output report file.")
    parser.add_argument("--min-digit-acc", type=float, default=0.92)
    parser.add_argument("--min-filled-precision", type=float, default=0.97)
    parser.add_argument("--max-latency-ms", type=float, default=120.0)
    return parser.parse_args()


def _fmt_check(name: str, value: float, ok: bool, target: float, cmp: str) -> str:
    status = "PASS" if ok else "FAIL"
    return f"- {status} {name}: {value:.4f} {cmp} {target:.4f}"


def main() -> int:
    args = parse_args()
    payload: Dict[str, Any] = json.loads(Path(args.benchmark).read_text())
    lines = ["# Benchmark Targets", ""]
    overall_ok = True

    for mode, metrics in payload.items():
        summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
        digit_acc = float(summary.get("digit_accuracy_on_filled_gt_mean", 0.0))
        precision = float(summary.get("filled_precision_mean", 0.0))
        latency = float(summary.get("latency_ms_mean", 99999.0))

        ok_digit = digit_acc >= args.min_digit_acc
        ok_precision = precision >= args.min_filled_precision
        ok_latency = latency <= args.max_latency_ms
        mode_ok = ok_digit and ok_precision and ok_latency
        overall_ok = overall_ok and mode_ok

        lines.append(f"## Mode: {mode}")
        lines.append(_fmt_check("digit_accuracy_on_filled_gt", digit_acc, ok_digit, args.min_digit_acc, ">="))
        lines.append(_fmt_check("filled_precision", precision, ok_precision, args.min_filled_precision, ">="))
        lines.append(_fmt_check("latency_ms", latency, ok_latency, args.max_latency_ms, "<="))
        lines.append("")

    lines.append(f"Overall: {'PASS' if overall_ok else 'FAIL'}")
    report = "\n".join(lines)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"\nReport written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
