from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import cv2
import numpy as np


TARGET_SIZE = 32
EMPTY_INK_THRESHOLD = 0.03
MIN_BEST_SCORE = 0.50
MIN_SCORE_MARGIN = 0.015


def _remove_border_artifacts(binary_cell: np.ndarray, margin: int = 1) -> np.ndarray:
    cleaned = binary_cell.copy()
    h, w = cleaned.shape
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        touches_border = x <= margin or y <= margin or (x + bw) >= (w - margin) or (y + bh) >= (h - margin)
        area = cv2.contourArea(cnt)
        thin_component = bw <= 2 or bh <= 2
        tiny_component = area < 0.02 * (h * w)
        if touches_border and (thin_component or tiny_component):
            cv2.drawContours(cleaned, [cnt], -1, 0, thickness=cv2.FILLED)
    return cleaned


def _inner_roi(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    pad_y = max(1, int(h * 0.2))
    pad_x = max(1, int(w * 0.2))
    roi = gray[pad_y : h - pad_y, pad_x : w - pad_x]
    return roi if roi.size != 0 else gray


def _normalize_cell_variants(cell_image: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
    roi = _inner_roi(gray)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    variants: List[np.ndarray] = []

    # Variant A: global Otsu.
    _, bw_otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(bw_otsu)

    # Variant B: adaptive threshold for uneven background/light.
    bw_adapt = cv2.adaptiveThreshold(
        roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    variants.append(bw_adapt)

    normalized: List[np.ndarray] = []
    kernel = np.ones((2, 2), dtype=np.uint8)
    for bw in variants:
        x = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        x = cv2.resize(x, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        _, x = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
        x = _remove_border_artifacts(x, margin=1)
        normalized.append(x)
    return normalized


def _is_empty(binary_cell: np.ndarray) -> bool:
    ink_ratio = float(np.count_nonzero(binary_cell)) / float(binary_cell.size)
    return ink_ratio < EMPTY_INK_THRESHOLD


def _extract_main_component(binary_cell: np.ndarray) -> np.ndarray:
    """
    Keep the largest connected component (likely the digit),
    then center it on a square canvas.
    """
    contours, _ = cv2.findContours(binary_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    h, w = binary_cell.shape
    min_area = 0.005 * (h * w)
    max_area = 0.60 * (h * w)
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        valid.append(cnt)
    if not valid:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    main = max(valid, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(main)
    digit = binary_cell[y : y + bh, x : x + bw]
    if digit.size == 0:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    side = int(max(bw, bh) * 1.35)
    side = max(side, 8)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - bh) // 2
    x0 = (side - bw) // 2
    canvas[y0 : y0 + bh, x0 : x0 + bw] = digit
    return cv2.resize(canvas, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=1)
def _digit_templates() -> Dict[int, List[np.ndarray]]:
    templates: Dict[int, List[np.ndarray]] = {}
    canvas_size = TARGET_SIZE
    for digit in range(1, 10):
        variants: List[np.ndarray] = []
        for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN):
            for scale in (0.85, 1.0, 1.15):
                for thickness in (2, 3):
                    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                    text = str(digit)
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                    x = (canvas_size - tw) // 2
                    y = (canvas_size + th) // 2
                    cv2.putText(canvas, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
                    _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
                    variants.append(canvas)
        templates[digit] = variants
    return templates


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = a.astype(np.float32).reshape(-1)
    vb = b.astype(np.float32).reshape(-1)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _shape_features(binary_cell: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(binary_cell > 0)
    if xs.size == 0 or ys.size == 0:
        return {"aspect_ratio": 0.0, "hole_count": 0.0, "ink_balance_tb": 0.0}

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    aspect_ratio = bw / float(bh)

    # Estimate holes from contour hierarchy.
    contours, hierarchy = cv2.findContours(binary_cell.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole_count = 0
    if hierarchy is not None and len(contours) > 0:
        for idx, cnt in enumerate(contours):
            parent = hierarchy[0][idx][3]
            if parent == -1:
                continue
            area = cv2.contourArea(cnt)
            if area > 4.0:
                hole_count += 1

    mid = binary_cell.shape[0] // 2
    top = float(np.count_nonzero(binary_cell[:mid, :]))
    bottom = float(np.count_nonzero(binary_cell[mid:, :]))
    denom = max(1.0, top + bottom)
    ink_balance_tb = min(top, bottom) / denom

    return {
        "aspect_ratio": aspect_ratio,
        "hole_count": float(hole_count),
        "ink_balance_tb": ink_balance_tb,
    }


def _apply_digit_shape_priors(best_score_by_digit: Dict[int, float], features: Dict[str, float]) -> Dict[int, float]:
    adjusted = dict(best_score_by_digit)
    aspect = features["aspect_ratio"]
    holes = int(features["hole_count"])
    balance = features["ink_balance_tb"]

    # Heuristics to reduce common 1 vs 8 confusions on Sudoku fonts.
    if holes >= 2:
        adjusted[8] += 0.08
        adjusted[1] -= 0.08
    elif holes == 0:
        adjusted[8] -= 0.10
        if aspect < 0.44:
            adjusted[1] += 0.05

    if aspect > 0.55:
        adjusted[1] -= 0.07
    if aspect < 0.40:
        adjusted[1] += 0.04

    if balance > 0.32:
        adjusted[8] += 0.04
    elif balance < 0.18:
        adjusted[8] -= 0.04

    return adjusted


def _match_digit(binary_cell: np.ndarray) -> Tuple[int, float, float, Dict[str, float]]:
    best_score_by_digit: Dict[int, float] = {}
    cell = (binary_cell > 0).astype(np.float32)

    for digit, templates in _digit_templates().items():
        best_for_digit = -1.0
        for tmpl in templates:
            template = (tmpl > 0).astype(np.float32)
            score = _cosine_similarity(cell, template)
            if score > best_for_digit:
                best_for_digit = score
        best_score_by_digit[digit] = best_for_digit

    features = _shape_features(binary_cell)
    adjusted_scores = _apply_digit_shape_priors(best_score_by_digit, features)
    ranked = sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return 0, 0.0, 0.0, features
    best_digit, best_score = ranked[0]
    second_best = ranked[1][1] if len(ranked) > 1 else 0.0
    return best_digit, best_score, second_best, features


def read_digit_with_confidence(cell_image: np.ndarray) -> tuple[int, float]:
    """
    Baseline OCR using OpenCV preprocessing + template matching.
    Returns (digit, confidence). digit=0 means empty/uncertain.
    """
    best_candidate: Tuple[int, float] = (0, 0.0)
    for binary_cell in _normalize_cell_variants(cell_image):
        if _is_empty(binary_cell):
            continue

        main_component = _extract_main_component(binary_cell)
        if _is_empty(main_component):
            continue

        digit, best_score, second_best, features = _match_digit(main_component)
        margin = best_score - second_best
        relaxed_accept = False
        aspect = features["aspect_ratio"]
        holes = features["hole_count"]

        # On this Sudoku font, true "4" usually has an internal hole and is wider.
        # A predicted "4" with no hole and narrow aspect is likely a "1".
        if digit == 4 and holes == 0 and aspect < 0.55:
            digit = 1
            best_score = max(best_score, second_best + 0.03)
            margin = max(0.03, best_score - second_best)

        # On this font, many "3" are confused as "6" when inner gap is weak.
        # If a predicted 6 has no inner hole, reinterpret it as 3.
        if digit == 6 and holes == 0 and 0.58 <= aspect <= 0.80:
            digit = 3
            best_score = max(best_score, second_best + 0.02)
            margin = max(0.02, best_score - second_best)

        # Very slender glyphs are often "1" on sudoku.com but can be confused with 3/4/7.
        if holes == 0 and aspect <= 0.37 and digit in (3, 4, 7):
            digit = 1
            best_score = max(best_score, second_best + 0.02)
            margin = max(0.02, best_score - second_best)

        if digit == 1 and aspect < 0.42 and best_score >= (MIN_BEST_SCORE - 0.05):
            relaxed_accept = True
        if digit == 8 and holes >= 1 and best_score >= (MIN_BEST_SCORE - 0.04):
            relaxed_accept = True
        if not relaxed_accept and (best_score < MIN_BEST_SCORE or margin < MIN_SCORE_MARGIN):
            continue

        confidence = best_score + 0.2 * margin
        if confidence > best_candidate[1]:
            best_candidate = (digit, confidence)

    return best_candidate


def read_digit(cell_image: np.ndarray) -> int:
    digit, _ = read_digit_with_confidence(cell_image)
    return digit
