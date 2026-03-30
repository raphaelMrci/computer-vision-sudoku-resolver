from __future__ import annotations

import io
import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from typing import List, Tuple

import numpy as np
from PIL import Image

from src.automation.interface import GridCalibration, fill_grid


API_URL = "http://localhost:8080/predict"
API_TIMEOUT_S = 120.0
MIN_CLUES = 17
COUNTDOWN_S = 3


def _log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


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
        actions.append((int(item[0]), int(item[1]), int(item[2])))
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


def main() -> int:
    t_main_start = time.perf_counter()
    _log("Starting live_solve client.")
    pyautogui = _require_pyautogui()
    _log("pyautogui ready.")

    _log("Interactive calibration started.")
    calibration = _interactive_calibration(pyautogui)
    _log("Interactive calibration done.")

    _log("Capture phase started.")
    _countdown(COUNTDOWN_S, "Capturing grid")
    t0 = time.perf_counter()
    image = _capture_region(pyautogui, calibration)
    capture_ms = (time.perf_counter() - t0) * 1000.0
    _log(f"Capture done ({capture_ms:.1f} ms).")

    _log("Calling solver API...")
    t0 = time.perf_counter()
    try:
        result = _predict_via_api(image)
    except RuntimeError as exc:
        print(str(exc))
        print("Quick checks:")
        print("- curl http://localhost:8080/health")
        print("- make docker-run  (or make api)")
        return 1
    infer_ms = (time.perf_counter() - t0) * 1000.0
    _log(
        f"API inference done (status={result.get('status')}, solved={result.get('solved')}, "
        f"clues={result.get('num_clues_detected')}, time={infer_ms:.1f} ms)."
    )

    print(json.dumps(result, indent=2))
    print(f"timings_ms capture={capture_ms:.1f} infer={infer_ms:.1f}")

    status = str(result.get("status", "unknown"))
    solved = bool(result.get("solved", False))
    clues = int(result.get("num_clues_detected", 0))
    actions = _parse_actions(result.get("actions"))
    centers = _build_cell_centers(calibration, image)
    print(f"parsed_actions={len(actions)} detected_cell_centers={len(centers)}")
    _log(f"Validation gate: status={status}, solved={solved}, clues={clues}, actions={len(actions)}")

    if not solved or not actions or status != "ok":
        print("Abort: puzzle not in a safe solved state.")
        return 1
    if clues < MIN_CLUES:
        print(f"Abort: detected clues {clues} < min-clues {MIN_CLUES}.")
        return 1

    _countdown(COUNTDOWN_S, "Auto-fill starts")
    if centers:
        k = next(iter(centers))
        fx, fy = centers[k]
        pyautogui.click(fx, fy, interval=0.06)
        time.sleep(0.08)

    t_fill = time.perf_counter()
    fill_grid(actions, calibration=calibration, cell_centers=centers, click_interval_s=0.06, key_interval_s=0.03)
    fill_ms = (time.perf_counter() - t_fill) * 1000.0
    total_ms = (time.perf_counter() - t_main_start) * 1000.0
    print(f"timings_ms fill_phase={fill_ms:.1f} total={total_ms:.1f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
