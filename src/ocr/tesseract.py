from __future__ import annotations

from functools import lru_cache
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


EMPTY_INK_THRESHOLD = 0.012
MIN_CONFIDENCE = 28.0


@lru_cache(maxsize=1)
def _has_tesseract() -> bool:
    try:
        import pytesseract  # type: ignore
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass
    return _resolve_tesseract_binary() is not None


@lru_cache(maxsize=1)
def _resolve_tesseract_binary() -> Optional[str]:
    candidates = [
        shutil.which("tesseract"),
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if p.exists() and p.is_file():
            return str(p)
    return None


def _prepare_cell(cell_image: np.ndarray, adaptive: bool = False, crop_ratio: float = 0.08) -> np.ndarray:
    gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    py = max(0, int(h * crop_ratio))
    px = max(0, int(w * crop_ratio))
    roi = gray[py : h - py, px : w - px]
    if roi.size == 0:
        roi = gray

    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    if adaptive:
        block = 15 if min(roi.shape[:2]) >= 15 else 11
        if block % 2 == 0:
            block += 1
        bw = cv2.adaptiveThreshold(
            roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            2,
        )
    else:
        _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))
    bw = cv2.resize(bw, (64, 64), interpolation=cv2.INTER_CUBIC)
    return bw


def _is_empty(binary_cell: np.ndarray) -> bool:
    ink = float(np.count_nonzero(binary_cell)) / float(binary_cell.size)
    return ink < EMPTY_INK_THRESHOLD


def _best_digit_from_data(data: dict) -> Tuple[int, float]:
    best_digit = 0
    best_conf = -1.0
    texts = data.get("text", [])
    confs = data.get("conf", [])
    for text, conf_str in zip(texts, confs):
        text = (text or "").strip()
        if not text or not text.isdigit():
            continue
        digit = int(text[0])
        if digit < 1 or digit > 9:
            continue
        try:
            conf = float(conf_str)
        except Exception:
            continue
        if conf > best_conf:
            best_conf = conf
            best_digit = digit
    if best_digit == 0:
        return 0, 0.0
    return best_digit, max(0.0, best_conf)


def _collect_digit_scores_from_data(data: dict) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    texts = data.get("text", [])
    confs = data.get("conf", [])
    for text, conf_str in zip(texts, confs):
        text = (text or "").strip()
        if not text or not text.isdigit():
            continue
        digit = int(text[0])
        if digit < 1 or digit > 9:
            continue
        try:
            conf = float(conf_str)
        except Exception:
            continue
        if conf > scores.get(digit, -1.0):
            scores[digit] = conf
    return scores


def _best_digit_from_tsv(tsv_text: str) -> Tuple[int, float]:
    lines = [line.strip("\n") for line in tsv_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0, 0.0
    header = lines[0].split("\t")
    try:
        text_idx = header.index("text")
        conf_idx = header.index("conf")
    except ValueError:
        return 0, 0.0
    best_digit = 0
    best_conf = -1.0
    for line in lines[1:]:
        cols = line.split("\t")
        if text_idx >= len(cols) or conf_idx >= len(cols):
            continue
        text = cols[text_idx].strip()
        if not text or not text.isdigit():
            continue
        digit = int(text[0])
        if digit < 1 or digit > 9:
            continue
        try:
            conf = float(cols[conf_idx])
        except Exception:
            continue
        if conf > best_conf:
            best_conf = conf
            best_digit = digit
    if best_digit == 0:
        return 0, 0.0
    return best_digit, max(0.0, best_conf)


def _collect_digit_scores_from_tsv(tsv_text: str) -> Dict[int, float]:
    lines = [line.strip("\n") for line in tsv_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    header = lines[0].split("\t")
    try:
        text_idx = header.index("text")
        conf_idx = header.index("conf")
    except ValueError:
        return {}
    scores: Dict[int, float] = {}
    for line in lines[1:]:
        cols = line.split("\t")
        if text_idx >= len(cols) or conf_idx >= len(cols):
            continue
        text = cols[text_idx].strip()
        if not text or not text.isdigit():
            continue
        digit = int(text[0])
        if digit < 1 or digit > 9:
            continue
        try:
            conf = float(cols[conf_idx])
        except Exception:
            continue
        if conf > scores.get(digit, -1.0):
            scores[digit] = conf
    return scores


def _read_with_tesseract_cli(binary_cell: np.ndarray) -> Tuple[int, float]:
    cmd = _resolve_tesseract_binary()
    if not cmd:
        return 0, 0.0
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ok = cv2.imwrite(tmp_path, binary_cell)
        if not ok:
            return 0, 0.0
        best_digit = 0
        best_conf = 0.0
        for psm in ("10", "13"):
            proc = subprocess.run(
                [
                    cmd,
                    tmp_path,
                    "stdout",
                    "--psm",
                    psm,
                    "--oem",
                    "3",
                    "-c",
                    "tessedit_char_whitelist=123456789",
                    "tsv",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if proc.returncode != 0:
                continue
            digit, conf = _best_digit_from_tsv(proc.stdout)
            if conf > best_conf:
                best_digit, best_conf = digit, conf
            if best_conf >= 70.0:
                break
        return best_digit, best_conf
    except Exception:
        return 0, 0.0
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _collect_scores_with_tesseract_cli(binary_cell: np.ndarray) -> Dict[int, float]:
    cmd = _resolve_tesseract_binary()
    if not cmd:
        return {}
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ok = cv2.imwrite(tmp_path, binary_cell)
        if not ok:
            return {}
        scores: Dict[int, float] = {}
        for psm in ("10", "13", "6"):
            proc = subprocess.run(
                [
                    cmd,
                    tmp_path,
                    "stdout",
                    "--psm",
                    psm,
                    "--oem",
                    "3",
                    "-c",
                    "tessedit_char_whitelist=123456789",
                    "tsv",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if proc.returncode != 0:
                continue
            partial = _collect_digit_scores_from_tsv(proc.stdout)
            for d, conf in partial.items():
                if conf > scores.get(d, -1.0):
                    scores[d] = conf
        return scores
    except Exception:
        return {}
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def read_digit_with_confidence_tesseract(cell_image: np.ndarray) -> Tuple[int, float]:
    if not _has_tesseract():
        return 0, 0.0

    bw_otsu = _prepare_cell(cell_image, adaptive=False)
    bw_adapt = _prepare_cell(cell_image, adaptive=True)
    if _is_empty(bw_otsu) and _is_empty(bw_adapt):
        return 0, 0.0

    try:
        import pytesseract  # type: ignore
        best_digit = 0
        best_conf = 0.0
        for bw in (bw_otsu, bw_adapt):
            for psm in ("10", "13"):
                config = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=123456789"
                data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config=config)
                digit, conf = _best_digit_from_data(data)
                if conf > best_conf:
                    best_digit, best_conf = digit, conf
                if best_conf >= 70.0:
                    break
            if best_conf >= 70.0:
                break
        digit, conf = best_digit, best_conf
    except Exception:
        d1, c1 = _read_with_tesseract_cli(bw_otsu)
        d2, c2 = _read_with_tesseract_cli(bw_adapt)
        if c2 > c1:
            digit, conf = d2, c2
        else:
            digit, conf = d1, c1
    if digit == 0 or conf < MIN_CONFIDENCE:
        return 0, conf / 100.0
    return digit, min(1.0, conf / 100.0)


def read_digit_candidates_tesseract_relaxed(
    cell_image: np.ndarray,
    min_confidence: float = 20.0,
    max_candidates: int = 3,
) -> List[Tuple[int, float]]:
    if not _has_tesseract():
        return []

    variants = [
        _prepare_cell(cell_image, adaptive=False, crop_ratio=0.00),
        _prepare_cell(cell_image, adaptive=True, crop_ratio=0.00),
        _prepare_cell(cell_image, adaptive=False, crop_ratio=0.06),
        _prepare_cell(cell_image, adaptive=True, crop_ratio=0.06),
        _prepare_cell(cell_image, adaptive=False, crop_ratio=0.12),
        _prepare_cell(cell_image, adaptive=True, crop_ratio=0.12),
    ]
    if all(_is_empty(v) for v in variants):
        return []

    merged: Dict[int, float] = {}
    try:
        import pytesseract  # type: ignore

        for bw in variants:
            for psm in ("10", "13", "6"):
                config = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=123456789"
                data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config=config)
                partial = _collect_digit_scores_from_data(data)
                for d, conf in partial.items():
                    if conf > merged.get(d, -1.0):
                        merged[d] = conf
    except Exception:
        for bw in variants:
            partial = _collect_scores_with_tesseract_cli(bw)
            for d, conf in partial.items():
                if conf > merged.get(d, -1.0):
                    merged[d] = conf

    ranked = sorted(
        [(d, c / 100.0) for d, c in merged.items() if c >= min_confidence],
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:max_candidates]
