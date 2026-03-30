from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class CaptureRegion:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_corners(cls, x1: int, y1: int, x2: int, y2: int) -> "CaptureRegion":
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width < 100 or height < 100:
            raise ValueError("Capture region is too small.")
        return cls(x=left, y=top, width=width, height=height)


@dataclass
class LoopConfig:
    region: CaptureRegion
    new_game_click: tuple[int, int]
    easy_click: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Sudoku screenshots in a loop: wait -> screenshot -> click New Game -> click Easy."
        )
    )
    parser.add_argument("--out-dir", default="data/raw_screenshots", help="Output folder for screenshots.")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds before each screenshot.")
    parser.add_argument("--click-delay", type=float, default=0.35, help="Delay between button clicks.")
    parser.add_argument("--prefix", default="sudoku", help="Screenshot filename prefix.")
    parser.add_argument("--countdown", action="store_true", help="Print a countdown before each capture.")
    parser.add_argument(
        "--save-config",
        default="data/raw_screenshots/capture_config.json",
        help="Path to save calibration metadata as JSON.",
    )
    return parser.parse_args()


def _require_pyautogui():
    try:
        import pyautogui  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyautogui is required. Run `make install`.") from exc
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
    return pyautogui


def _wait_enter(prompt: str) -> None:
    print(prompt)
    input()


def _calibrate(pyautogui) -> LoopConfig:
    _wait_enter("Move mouse to TOP-LEFT of screenshot zone, then press Enter.")
    x1, y1 = pyautogui.position()
    print(f"Captured top-left: ({x1}, {y1})")

    _wait_enter("Move mouse to BOTTOM-RIGHT of screenshot zone, then press Enter.")
    x2, y2 = pyautogui.position()
    print(f"Captured bottom-right: ({x2}, {y2})")
    region = CaptureRegion.from_corners(x1, y1, x2, y2)
    print(f"Region => x={region.x} y={region.y} w={region.width} h={region.height}")

    _wait_enter("Move mouse to NEW GAME button center, then press Enter.")
    ng_x, ng_y = pyautogui.position()
    print(f"Captured new game click: ({ng_x}, {ng_y})")

    _wait_enter("Move mouse to EASY button center, then press Enter.")
    easy_x, easy_y = pyautogui.position()
    print(f"Captured easy click: ({easy_x}, {easy_y})")

    return LoopConfig(region=region, new_game_click=(ng_x, ng_y), easy_click=(easy_x, easy_y))


def _capture(pyautogui, region: CaptureRegion) -> np.ndarray:
    image = pyautogui.screenshot(region=(region.x, region.y, region.width, region.height))
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return np.array(image.convert("RGB"))


def _countdown(seconds: float) -> None:
    whole = int(seconds)
    if whole <= 0:
        return
    for i in range(whole, 0, -1):
        print(f"Capture in {i}s...")
        time.sleep(1)
    remaining = seconds - whole
    if remaining > 0:
        time.sleep(remaining)


def main() -> int:
    args = parse_args()
    pyautogui = _require_pyautogui()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _calibrate(pyautogui)
    config_path = Path(args.save_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(asdict(config), indent=2))
    print(f"Saved calibration config to: {config_path}")

    print("Starting capture loop. Press Ctrl+C to stop.")
    index = 0
    try:
        while True:
            if args.countdown:
                _countdown(args.interval)
            else:
                time.sleep(args.interval)

            frame = _capture(pyautogui, config.region)
            timestamp_ms = int(time.time() * 1000)
            out_path = out_dir / f"{args.prefix}_{index:06d}_{timestamp_ms}.png"
            Image.fromarray(frame).save(out_path)
            print(f"[{index}] saved {out_path}")

            pyautogui.click(config.new_game_click[0], config.new_game_click[1], interval=args.click_delay)
            time.sleep(args.click_delay)
            pyautogui.click(config.easy_click[0], config.easy_click[1], interval=args.click_delay)
            index += 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
        print(f"Total screenshots saved: {index}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
