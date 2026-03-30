from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class CellBox:
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int


def detect_cells(image: np.ndarray) -> List[CellBox]:
    """
    Detect the global Sudoku grid bounding box, then split it into 9x9 cells.

    This is a classical CV fallback before integrating a fine-tuned detector.
    """
    x, y, side = _detect_grid_square(image)
    cell_w = side // 9
    cell_h = side // 9

    boxes: List[CellBox] = []
    for row in range(9):
        for col in range(9):
            x1 = x + col * cell_w
            y1 = y + row * cell_h
            x2 = x + side if col == 8 else x + (col + 1) * cell_w
            y2 = y + side if row == 8 else y + (row + 1) * cell_h
            boxes.append(CellBox(row=row, col=col, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def crop_cell(image: np.ndarray, box: CellBox) -> np.ndarray:
    return image[box.y1 : box.y2, box.x1 : box.x2]


def _detect_grid_square(image: np.ndarray) -> Tuple[int, int, int]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        side = min(width, height)
        return 0, 0, side

    image_area = width * height
    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.2 * image_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h) if h > 0 else 0.0
        ratio_penalty = abs(1.0 - ratio)
        if ratio_penalty > 0.35:
            continue
        score = area - (ratio_penalty * image_area * 0.2)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        side = min(width, height)
        return 0, 0, side

    x, y, w, h = best
    side = min(w, h)
    cx = x + (w // 2)
    cy = y + (h // 2)
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    if x0 + side > width:
        x0 = width - side
    if y0 + side > height:
        y0 = height - side
    return int(x0), int(y0), int(side)
