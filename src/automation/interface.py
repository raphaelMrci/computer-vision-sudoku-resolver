from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GridCalibration:
    x: int
    y: int
    size: int

    @classmethod
    def from_corners(cls, x1: int, y1: int, x2: int, y2: int) -> "GridCalibration":
        left = min(x1, x2)
        top = min(y1, y2)
        side = min(abs(x2 - x1), abs(y2 - y1))
        if side < 90:
            raise ValueError("Grid selection is too small.")
        return cls(x=left, y=top, size=side)

    def cell_center(self, row: int, col: int) -> Tuple[int, int]:
        cell = self.size / 9.0
        cx = int(self.x + (col + 0.5) * cell)
        cy = int(self.y + (row + 0.5) * cell)
        return cx, cy


def fill_grid(
    actions: List[Tuple[int, int, int]],
    calibration: GridCalibration | None = None,
    cell_centers: Dict[Tuple[int, int], Tuple[int, int]] | None = None,
    click_interval_s: float = 0.03,
    key_interval_s: float = 0.015,
) -> None:
    """
    Fill Sudoku grid using keyboard automation.

    If no calibration is provided, this function logs actions only.
    """
    if calibration is None and cell_centers is None:
        for row, col, value in actions:
            print(f"[automation] set cell ({row}, {col}) = {value}")
        return

    try:
        import pyautogui  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyautogui is required for live browser automation.") from exc

    pyautogui.PAUSE = 0.0
    pyautogui.FAILSAFE = True

    for row, col, value in actions:
        if cell_centers is not None and (row, col) in cell_centers:
            x, y = cell_centers[(row, col)]
        elif calibration is not None:
            x, y = calibration.cell_center(row, col)
        else:
            print(f"[automation] missing target for ({row}, {col}), skipping")
            continue
        pyautogui.click(x, y, interval=click_interval_s)
        pyautogui.press(str(value), interval=key_interval_s)
