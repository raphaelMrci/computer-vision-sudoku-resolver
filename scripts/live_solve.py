from __future__ import annotations

import argparse
import io
import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from src.automation.interface import GridCalibration, fill_grid


API_URL = "http://localhost:8080/predict"
API_TIMEOUT_S = 120.0
MIN_CLUES = 17
COUNTDOWN_S = 3
MAX_CONSECUTIVE_FAILURES = 10
DEFAULT_CONFIG_PATH = Path("data/live_solve_config.json")
STATS_LOG_PATH = Path("data/live_solve_stats.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Sudoku client.")
    parser.add_argument("--loop", action="store_true", help="Loop forever.")
    parser.add_argument("--save-config", action="store_true", help="Save clicks to config file.")
    return parser.parse_args()


def _require_pyautogui():
    try:
        import pyautogui  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyautogui is required. Install dependencies with `make install`.") from exc
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
    return pyautogui


def _pause(seconds: int) -> None:
    time.sleep(seconds)


def _capture_region(pyautogui, calibration: GridCalibration) -> np.ndarray:
    img = pyautogui.screenshot(region=(calibration.x, calibration.y, calibration.size, calibration.size))
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return np.array(img.convert("RGB"))


def _interactive_calibration(pyautogui) -> GridCalibration:
    print("top-left, enter")
    input()
    x1, y1 = pyautogui.position()

    print("bottom-right, enter")
    input()
    x2, y2 = pyautogui.position()

    return GridCalibration.from_corners(x1, y1, x2, y2)


def _capture_point(pyautogui, prompt: str) -> Tuple[int, int]:
    print(prompt)
    input()
    x, y = pyautogui.position()
    return int(x), int(y)


def _load_config(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _save_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _append_stats_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def _parse_actions(raw_actions: object) -> List[Tuple[int, int, int]]:
    actions: List[Tuple[int, int, int]] = []
    if not isinstance(raw_actions, list):
        return actions
    for item in raw_actions:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        actions.append((int(item[0]), int(item[1]), int(item[2])))
    return actions


def _parse_grid(raw_grid: object) -> List[List[int]]:
    if not isinstance(raw_grid, list) or len(raw_grid) != 9:
        return []
    out: List[List[int]] = []
    for row in raw_grid:
        if not isinstance(row, list) or len(row) != 9:
            return []
        out.append([int(v) for v in row])
    return out


def _base_metrics(ts: str, status: str, clues: int, actions_count: int) -> dict:
    return {
        "ts": ts,
        "status": status,
        "clues": clues,
        "actions": actions_count,
    }


def _finalize_metrics(metrics: dict, capture_ms: float, infer_ms: float, fill_ms: float, total_ms: float) -> dict:
    metrics["capture_ms"] = round(capture_ms, 1)
    metrics["infer_ms"] = round(infer_ms, 1)
    metrics["fill_ms"] = round(fill_ms, 1)
    metrics["total_ms"] = round(total_ms, 1)
    return metrics


def _actions_from_solved_grid(solved_grid: List[List[int]]) -> List[Tuple[int, int, int]]:
    actions: List[Tuple[int, int, int]] = []
    for r in range(9):
        for c in range(9):
            value = solved_grid[r][c]
            if 1 <= value <= 9:
                actions.append((r, c, value))
    return actions


def _encode_multipart_formdata(field_name: str, filename: str, data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"----sudoku-live-{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
        data,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), boundary


def _predict_via_api(image: np.ndarray) -> dict:
    pil = Image.fromarray(image)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    body, boundary = _encode_multipart_formdata("file", "capture.png", buffer.getvalue(), "image/png")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_S) as resp:
            payload = resp.read()
            code = resp.getcode()
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"API timeout after {API_TIMEOUT_S:.0f}s. Make sure solver API is running: {API_URL}"
        ) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API unreachable: {exc}") from exc
    if code != 200:
        raise RuntimeError(f"API returned HTTP {code}: {payload.decode('utf-8', errors='replace')}")
    return json.loads(payload.decode("utf-8"))


def _build_cell_centers(calibration: GridCalibration, image: np.ndarray) -> dict[Tuple[int, int], Tuple[int, int]]:
    from src.detection.interface import detect_cells

    centers: dict[Tuple[int, int], Tuple[int, int]] = {}
    for box in detect_cells(image):
        cx = calibration.x + ((box.x1 + box.x2) // 2)
        cy = calibration.y + ((box.y1 + box.y2) // 2)
        centers[(box.row, box.col)] = (cx, cy)
    return centers


def _start_new_extreme_game(
    pyautogui,
    new_game_pos: Tuple[int, int],
    extreme_pos: Tuple[int, int],
    pre_delay_s: float = 0.0,
) -> None:
    nx, ny = new_game_pos
    ex, ey = extreme_pos
    if pre_delay_s > 0:
        time.sleep(pre_delay_s)
    pyautogui.click(nx, ny, interval=0.10)
    time.sleep(0.35)
    pyautogui.click(ex, ey, interval=0.10)
    time.sleep(0.50)


def _run_single_attempt(pyautogui, calibration: GridCalibration) -> Tuple[bool, str, dict]:
    t_main_start = time.perf_counter()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    capture_ms = 0.0
    infer_ms = 0.0
    fill_ms = 0.0
    status = "unknown"
    clues = 0
    actions_count = 0
    print("capture...")
    _pause(COUNTDOWN_S)
    t0 = time.perf_counter()
    image = _capture_region(pyautogui, calibration)
    capture_ms = (time.perf_counter() - t0) * 1000.0

    print("processing...")
    t0 = time.perf_counter()
    try:
        result = _predict_via_api(image)
    except RuntimeError as exc:
        print(str(exc))
        print("api running? try: make api")
        total_ms = (time.perf_counter() - t_main_start) * 1000.0
        metrics = _base_metrics(ts, status, clues, actions_count)
        metrics["ok"] = False
        metrics["reason"] = "api_error"
        return False, "api_error", _finalize_metrics(metrics, capture_ms, infer_ms, fill_ms, total_ms)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    print(json.dumps(result, indent=2))

    status = str(result.get("status", "unknown"))
    solved = bool(result.get("solved", False))
    clues = int(result.get("num_clues_detected", 0))
    solved_grid = _parse_grid(result.get("solved_grid"))
    actions = _actions_from_solved_grid(solved_grid) if solved_grid else _parse_actions(result.get("actions"))
    actions_count = len(actions)
    centers = _build_cell_centers(calibration, image)

    if not solved or not actions or status != "ok":
        print("didn't solve, skipping fill.")
        total_ms = (time.perf_counter() - t_main_start) * 1000.0
        metrics = _base_metrics(ts, status, clues, actions_count)
        metrics["ok"] = False
        metrics["reason"] = "unsolved_or_unsafe"
        return False, "unsolved_or_unsafe", _finalize_metrics(metrics, capture_ms, infer_ms, fill_ms, total_ms)
    if clues < MIN_CLUES:
        print(f"too few clues ({clues}), skipping.")
        total_ms = (time.perf_counter() - t_main_start) * 1000.0
        metrics = _base_metrics(ts, status, clues, actions_count)
        metrics["ok"] = False
        metrics["reason"] = "too_few_clues"
        return False, "too_few_clues", _finalize_metrics(metrics, capture_ms, infer_ms, fill_ms, total_ms)

    print("typing...")
    if actions:
        first_row, first_col, _ = actions[0]
        if (first_row, first_col) in centers:
            fx, fy = centers[(first_row, first_col)]
            pyautogui.click(fx, fy, interval=0.06)
            time.sleep(0.08)
    elif centers:
        k = next(iter(centers))
        fx, fy = centers[k]
        pyautogui.click(fx, fy, interval=0.06)
        time.sleep(0.08)

    t_fill = time.perf_counter()
    fill_grid(actions, calibration=calibration, cell_centers=centers, click_interval_s=0.09, key_interval_s=0.05)
    fill_ms = (time.perf_counter() - t_fill) * 1000.0
    total_ms = (time.perf_counter() - t_main_start) * 1000.0
    print("done.")
    metrics = _base_metrics(ts, status, clues, actions_count)
    metrics["ok"] = True
    metrics["reason"] = "ok"
    return True, "ok", _finalize_metrics(metrics, capture_ms, infer_ms, fill_ms, total_ms)


def main() -> int:
    args = parse_args()
    pyautogui = _require_pyautogui()
    print("mouse top-left corner = stop")

    cfg = _load_config(DEFAULT_CONFIG_PATH)
    if cfg is not None:
        try:
            grid = cfg["grid"]
            calibration = GridCalibration(
                x=int(grid["x"]),
                y=int(grid["y"]),
                size=int(grid["size"]),
            )
            print("using saved grid box")
        except Exception:
            cfg = None
    if cfg is None:
        calibration = _interactive_calibration(pyautogui)
        cfg = {
            "grid": {"x": calibration.x, "y": calibration.y, "size": calibration.size},
        }

    if args.save_config:
        new_game_success_pos = _capture_point(pyautogui, "new game (win screen), enter")
        extreme_success_pos = _capture_point(pyautogui, "extreme (win screen), enter")
        new_game_failed_pos = _capture_point(pyautogui, "new game (lose screen), enter")
        extreme_failed_pos = _capture_point(pyautogui, "extreme (lose screen), enter")
        cfg["loop_buttons"] = {
            "success": {"new_game": list(new_game_success_pos), "extreme": list(extreme_success_pos)},
            "failed": {"new_game": list(new_game_failed_pos), "extreme": list(extreme_failed_pos)},
        }
        _save_config(DEFAULT_CONFIG_PATH, cfg)
        print(f"saved {DEFAULT_CONFIG_PATH}")
        return 0

    if not args.loop:
        ok, reason, metrics = _run_single_attempt(pyautogui, calibration)
        metrics["mode"] = "single"
        _append_stats_event(STATS_LOG_PATH, metrics)
        return 0 if ok else 1

    loop_cfg = cfg.get("loop_buttons", {}) if isinstance(cfg, dict) else {}
    try:
        s_cfg = loop_cfg["success"]
        f_cfg = loop_cfg["failed"]
        new_game_success_pos = (int(s_cfg["new_game"][0]), int(s_cfg["new_game"][1]))
        extreme_success_pos = (int(s_cfg["extreme"][0]), int(s_cfg["extreme"][1]))
        new_game_failed_pos = (int(f_cfg["new_game"][0]), int(f_cfg["new_game"][1]))
        extreme_failed_pos = (int(f_cfg["extreme"][0]), int(f_cfg["extreme"][1]))
        print("loop buttons ok")
    except Exception:
        print("run make config first")
        return 1

    total = 0
    success = 0
    failed = 0
    consecutive_failures = 0
    try:
        while True:
            game_idx = total + 1
            print(f"game {game_idx}")
            ok, reason, metrics = _run_single_attempt(pyautogui, calibration)
            metrics["mode"] = "loop"
            metrics["game_idx"] = game_idx
            _append_stats_event(STATS_LOG_PATH, metrics)
            total += 1
            if ok:
                success += 1
                consecutive_failures = 0
            else:
                failed += 1
                consecutive_failures += 1
            rate = (100.0 * success / total) if total else 0.0
            print(f"n={total} win%={rate:.0f} last={reason} streak_fail={consecutive_failures}")

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"bailing after {MAX_CONSECUTIVE_FAILURES} fails in a row")
                break

            if ok:
                _start_new_extreme_game(
                    pyautogui,
                    new_game_success_pos,
                    extreme_success_pos,
                    pre_delay_s=1.4,
                )
            else:
                _start_new_extreme_game(pyautogui, new_game_failed_pos, extreme_failed_pos)
            time.sleep(0.8)
    except KeyboardInterrupt:
        print("stopped (ctrl+c)")
    except Exception as exc:  # pragma: no cover
        print(f"crash: {exc}")

    rate = (100.0 * success / total) if total else 0.0
    print(f"done. n={total} win%={rate:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
