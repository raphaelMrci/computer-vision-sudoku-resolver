from __future__ import annotations

import argparse
import json
import time
from typing import List, Tuple

import numpy as np
from PIL import Image

from src.automation.interface import GridCalibration, fill_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Sudoku solve (simple flow, Tesseract OCR only).")
    parser.add_argument("--ocr-mode", default="tesseract", help=argparse.SUPPRESS)
    parser.add_argument("--rescue-attempts", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--x", type=int, help="Grid top-left x in screen coordinates.")
    parser.add_argument("--y", type=int, help="Grid top-left y in screen coordinates.")
    parser.add_argument("--size", type=int, help="Grid side length in pixels.")
    parser.add_argument("--min-clues", type=int, default=17, help="Minimum detected clues to allow auto-fill.")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown before capture/fill.")
    parser.add_argument("--allow-relaxed", action="store_true", help="Allow auto-fill when status is ok_relaxed.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Do not click/type, print actions only.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stage logs.")
    return parser.parse_args()


def _require_pyautogui():
    try:
        import pyautogui  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyautogui is required. Install dependencies with `make install`.") from exc
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
    return pyautogui


def _countdown(seconds: int, message: str) -> None:
    for i in range(seconds, 0, -1):
        print(f"{message} in {i}...")
        time.sleep(1)


def _log(enabled: bool, message: str) -> None:
    if not enabled:
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


def _capture_region(pyautogui, calibration: GridCalibration) -> np.ndarray:
    img = pyautogui.screenshot(region=(calibration.x, calibration.y, calibration.size, calibration.size))
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return np.array(img.convert("RGB"))


def _interactive_calibration(pyautogui) -> GridCalibration:
    print("Move mouse to TOP-LEFT corner of Sudoku grid, then press Enter.")
    input()
    x1, y1 = pyautogui.position()
    print(f"Captured top-left: ({x1}, {y1})")

    print("Move mouse to BOTTOM-RIGHT corner of Sudoku grid, then press Enter.")
    input()
    x2, y2 = pyautogui.position()
    print(f"Captured bottom-right: ({x2}, {y2})")

    calibration = GridCalibration.from_corners(x1, y1, x2, y2)
    print(f"Calibration => x={calibration.x}, y={calibration.y}, size={calibration.size}")
    return calibration


def _parse_actions(raw_actions: object) -> List[Tuple[int, int, int]]:
    actions: List[Tuple[int, int, int]] = []
    if not isinstance(raw_actions, list):
        return actions
    for item in raw_actions:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        row, col, value = int(item[0]), int(item[1]), int(item[2])
        actions.append((row, col, value))
    return actions


def _parse_confidence_grid(raw_grid: object) -> List[List[float]]:
    if not isinstance(raw_grid, list) or len(raw_grid) != 9:
        return [[0.0 for _ in range(9)] for _ in range(9)]
    parsed: List[List[float]] = []
    for row in raw_grid:
        if not isinstance(row, list) or len(row) != 9:
            return [[0.0 for _ in range(9)] for _ in range(9)]
        parsed.append([float(v) for v in row])
    return parsed


def _parse_grid(raw_grid: object) -> List[List[int]]:
    if not isinstance(raw_grid, list) or len(raw_grid) != 9:
        return [[0 for _ in range(9)] for _ in range(9)]
    parsed: List[List[int]] = []
    for row in raw_grid:
        if not isinstance(row, list) or len(row) != 9:
            return [[0 for _ in range(9)] for _ in range(9)]
        parsed.append([int(v) for v in row])
    return parsed


def _reconcile_givens(
    primary: dict,
    verify: dict,
) -> Tuple[List[List[int]], List[List[float]], int]:
    primary_grid = _parse_grid(primary.get("initial_grid"))
    primary_conf = _parse_confidence_grid(primary.get("confidence_grid"))
    verify_grid = _parse_grid(verify.get("initial_grid"))
    verify_conf = _parse_confidence_grid(verify.get("confidence_grid"))

    merged_grid = [row[:] for row in primary_grid]
    merged_conf = [row[:] for row in primary_conf]
    changes = 0
    for r in range(9):
        for c in range(9):
            p = primary_grid[r][c]
            pc = primary_conf[r][c]
            v = verify_grid[r][c]
            vc = verify_conf[r][c]
            # Recover likely missed givens from a second OCR pass.
            if p == 0 and v != 0 and vc >= 0.72:
                merged_grid[r][c] = v
                merged_conf[r][c] = vc
                changes += 1
            # Resolve disagreements by trusting clearly stronger confidence.
            elif p != 0 and v != 0 and p != v and vc >= pc + 0.10 and vc >= 0.86:
                merged_grid[r][c] = v
                merged_conf[r][c] = vc
                changes += 1
    return merged_grid, merged_conf, changes


def _result_rank(result: dict) -> Tuple[int, int, int]:
    solved = 1 if bool(result.get("solved", False)) else 0
    unique = 1 if int(result.get("solution_count_capped", 0)) == 1 else 0
    clues = int(result.get("num_clues_detected", 0))
    return solved, unique, clues


def _build_cell_centers(calibration: GridCalibration, image: np.ndarray) -> dict[Tuple[int, int], Tuple[int, int]]:
    try:
        from src.detection.interface import detect_cells
    except ModuleNotFoundError as exc:  # pragma: no cover
        missing = getattr(exc, "name", "dependency")
        raise RuntimeError(
            f"Missing dependency: {missing}. Install project dependencies first with `make install`."
        ) from exc

    centers: dict[Tuple[int, int], Tuple[int, int]] = {}
    for box in detect_cells(image):
        cx = calibration.x + ((box.x1 + box.x2) // 2)
        cy = calibration.y + ((box.y1 + box.y2) // 2)
        centers[(box.row, box.col)] = (cx, cy)
    return centers


def main() -> int:
    t_main_start = time.perf_counter()
    args = parse_args()
    verbose = not args.quiet
    _log(verbose, "Starting live_solve.")
    pyautogui = _require_pyautogui()
    _log(verbose, "pyautogui ready.")
    try:
        from src.pipeline import run_pipeline
        from src.pipeline import run_pipeline_from_grid
        from src.ocr.tesseract import _has_tesseract
    except ModuleNotFoundError as exc:  # pragma: no cover
        missing = getattr(exc, "name", "dependency")
        print(f"Missing dependency: {missing}")
        print("Install project dependencies first: `make install`")
        return 1

    ocr_mode = "tesseract"
    if str(args.ocr_mode).lower() != "tesseract":
        _log(verbose, f"Ignoring --ocr-mode={args.ocr_mode}; forced to tesseract.")
    _log(verbose, f"OCR mode fixed: {ocr_mode}")
    if not _has_tesseract():
        print("Abort: tesseract backend not found.")
        print("Install tesseract binary or add it to PATH, then retry.")
        return 1
    if args.x is not None and args.y is not None and args.size is not None:
        calibration = GridCalibration(x=args.x, y=args.y, size=args.size)
        _log(verbose, f"Calibration provided by CLI: x={args.x}, y={args.y}, size={args.size}")
    else:
        _log(verbose, "Interactive calibration started.")
        calibration = _interactive_calibration(pyautogui)
        _log(verbose, "Interactive calibration done.")

    _log(verbose, "Capture phase started.")
    _countdown(args.countdown, "Capturing grid")
    t0 = time.perf_counter()
    image = _capture_region(pyautogui, calibration)
    capture_ms = (time.perf_counter() - t0) * 1000.0
    _log(verbose, f"Capture done ({capture_ms:.1f} ms).")

    _log(verbose, "Running primary inference...")
    t0 = time.perf_counter()
    result = run_pipeline(image, ocr_mode=ocr_mode)
    first_infer_ms = (time.perf_counter() - t0) * 1000.0
    _log(
        verbose,
        "Primary inference done "
        f"(status={result.get('status')}, solved={result.get('solved')}, clues={result.get('num_clues_detected')}, "
        f"time={first_infer_ms:.1f} ms).",
    )

    # If solve is ambiguous/unsolved, recover missing givens before aborting.
    for verify_idx in range(2):
        solved = bool(result.get("solved", False))
        unique = int(result.get("solution_count_capped", 0)) == 1
        if solved and unique:
            break
        _log(verbose, f"Given-verification OCR pass #{verify_idx + 1}...")
        verify_image = _capture_region(pyautogui, calibration)
        verify = run_pipeline(verify_image, ocr_mode=ocr_mode)
        merged_grid, merged_conf, merged_changes = _reconcile_givens(result, verify)
        if merged_changes > 0:
            _log(verbose, f"Recovered/updated {merged_changes} givens; re-solving.")
            result = run_pipeline_from_grid(merged_grid, merged_conf, ocr_mode=ocr_mode)
            image = verify_image
            continue
        # No merge diff: still keep better direct pass if it improves.
        if _result_rank(verify) > _result_rank(result):
            _log(verbose, "Verification pass produced a better candidate.")
            result = verify
            image = verify_image
        else:
            _log(verbose, "Verification pass did not improve result.")
    print(json.dumps(result, indent=2))
    print(
        "timings_ms "
        f"capture={capture_ms:.1f} "
        f"first_infer={first_infer_ms:.1f} "
        "rescue_infer_total=0.0 "
        "rescue_runs=0"
    )

    status = result.get("status")
    clues = int(result.get("num_clues_detected", 0))
    solved = bool(result.get("solved", False))
    actions = _parse_actions(result.get("actions"))
    cell_centers = _build_cell_centers(calibration, image)
    print(f"parsed_actions={len(actions)} detected_cell_centers={len(cell_centers)}")
    _log(
        verbose,
        f"Validation gate: status={status}, solved={solved}, clues={clues}, actions={len(actions)}",
    )

    allowed_status = {"ok", "ok_relaxed"} if args.allow_relaxed else {"ok"}
    if not solved or not actions or status not in allowed_status:
        print("Abort: puzzle not in a safe solved state.")
        _log(verbose, "Abort at safety gate: unsafe solved state.")
        return 1

    if clues < args.min_clues:
        print(f"Abort: detected clues {clues} < min-clues {args.min_clues}.")
        _log(verbose, "Abort at min-clues gate.")
        return 1

    if args.dry_run:
        _log(verbose, "Dry-run enabled; printing actions only.")
        fill_grid(actions, calibration=None)
        return 0

    if not args.yes:
        print("Ready to auto-fill browser now. Type 'yes' to continue:")
        if input().strip().lower() != "yes":
            print("Cancelled.")
            _log(verbose, "Cancelled by user before autofill.")
            return 0

    _log(verbose, "Autofill phase started.")
    _countdown(args.countdown, "Auto-fill starts")
    # Prime browser focus before typing.
    if cell_centers:
        first_key = next(iter(cell_centers.keys()))
        fx, fy = cell_centers[first_key]
        pyautogui.click(fx, fy, interval=0.06)
        time.sleep(0.08)
    t_fill_start = time.perf_counter()
    fill_grid(actions, calibration=calibration, cell_centers=cell_centers, click_interval_s=0.06, key_interval_s=0.03)
    _log(verbose, "Autofill completed.")

    total_ms = (time.perf_counter() - t_main_start) * 1000.0
    fill_ms = (time.perf_counter() - t_fill_start) * 1000.0
    print(f"timings_ms fill_phase={fill_ms:.1f} total={total_ms:.1f}")
    _log(verbose, f"Done. fill_phase={fill_ms:.1f}ms total={total_ms:.1f}ms")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
