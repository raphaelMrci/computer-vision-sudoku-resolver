from __future__ import annotations

from functools import lru_cache
from typing import Dict

import cv2
import numpy as np


TARGET_SIZE = 32
EMPTY_INK_THRESHOLD = 0.025
MIN_BEST_SCORE = 0.46
MIN_SCORE_MARGIN = 0.01


def _normalize_cell(cell_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    pad_y = max(1, int(h * 0.2))
    pad_x = max(1, int(w * 0.2))
    roi = gray[pad_y : h - pad_y, pad_x : w - pad_x]
    if roi.size == 0:
        roi = gray

    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    bw = cv2.resize(bw, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)
    return bw


def _is_empty(binary_cell: np.ndarray) -> bool:
    return (float(np.count_nonzero(binary_cell)) / float(binary_cell.size)) < EMPTY_INK_THRESHOLD


@lru_cache(maxsize=1)
def _digit_templates() -> Dict[int, np.ndarray]:
    templates: Dict[int, np.ndarray] = {}
    for digit in range(1, 10):
        canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        text = str(digit)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (TARGET_SIZE - tw) // 2
        y = (TARGET_SIZE + th) // 2
        cv2.putText(canvas, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        templates[digit] = canvas
    return templates


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = a.astype(np.float32).reshape(-1)
    vb = b.astype(np.float32).reshape(-1)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def read_digit_with_confidence_basic(cell_image: np.ndarray) -> tuple[int, float]:
    binary = _normalize_cell(cell_image)
    if _is_empty(binary):
        return 0, 0.0

    cell = (binary > 0).astype(np.float32)
    best_digit = 0
    best_score = -1.0
    second_best = -1.0
    for digit, tmpl in _digit_templates().items():
        score = _cosine_similarity(cell, (tmpl > 0).astype(np.float32))
        if score > best_score:
            second_best = best_score
            best_score = score
            best_digit = digit
        elif score > second_best:
            second_best = score

    margin = best_score - second_best
    if best_score < MIN_BEST_SCORE or margin < MIN_SCORE_MARGIN:
        return 0, max(0.0, best_score)
    return best_digit, best_score + 0.2 * margin
