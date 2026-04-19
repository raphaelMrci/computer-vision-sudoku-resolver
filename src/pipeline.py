from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection.interface import CellBox, crop_cell, detect_cells
from src.ocr.tesseract import read_digit_candidates_tesseract_relaxed, read_digit_with_confidence_tesseract
from src.solver.backtracking import count_solutions_with_budget, solve_sudoku_with_budget

MIN_GIVEN_CONFIDENCE = 0.60
UNCERTAIN_CLUE_THRESHOLD = 0.72
CONFIDENT_CLUE_THRESHOLD = 0.86


def _build_grid(
    image: np.ndarray,
    cell_boxes: List[CellBox],
    ocr_mode: str,
) -> Tuple[List[List[int]], List[List[float]]]:
    _ = ocr_mode
    grid = [[0 for _ in range(9)] for _ in range(9)]
    confidence = [[0.0 for _ in range(9)] for _ in range(9)]
    for box in cell_boxes:
        cell = crop_cell(image, box)
        digit, conf = read_digit_with_confidence_tesseract(cell)
        grid[box.row][box.col] = digit
        confidence[box.row][box.col] = conf
    return grid, confidence


def _actions_from_solution(initial: List[List[int]], solved: List[List[int]]) -> List[Tuple[int, int, int]]:
    actions: List[Tuple[int, int, int]] = []
    for row in range(9):
        for col in range(9):
            if initial[row][col] == 0:
                actions.append((row, col, solved[row][col]))
    return actions


def _count_clues(grid: List[List[int]]) -> int:
    return sum(1 for row in grid for value in row if value != 0)


def _confidence_stats(grid: List[List[int]], confidence: List[List[float]]) -> Dict[str, float]:
    values: List[float] = []
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                values.append(confidence[r][c])
    if not values:
        return {"ocr_confidence_mean": 0.0, "ocr_confidence_min": 0.0, "ocr_confidence_max": 0.0}
    return {
        "ocr_confidence_mean": float(np.mean(values)),
        "ocr_confidence_min": float(np.min(values)),
        "ocr_confidence_max": float(np.max(values)),
        "ocr_confidence_std": float(np.std(values)),
    }


def _uncertainty_stats(grid: List[List[int]], confidence: List[List[float]]) -> Dict[str, float]:
    clue_count = 0
    uncertain = 0
    confident = 0
    quality_sum = 0.0

    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                continue
            clue_count += 1
            conf = confidence[r][c]
            quality_sum += conf
            if conf < UNCERTAIN_CLUE_THRESHOLD:
                uncertain += 1
            if conf >= CONFIDENT_CLUE_THRESHOLD:
                confident += 1

    if clue_count == 0:
        return {
            "num_uncertain_clues": 0,
            "num_confident_clues": 0,
            "clue_quality_score": 0.0,
        }

    return {
        "num_uncertain_clues": uncertain,
        "num_confident_clues": confident,
        "clue_quality_score": float(quality_sum / clue_count),
    }


def _uncertain_cells(grid: List[List[int]], confidence: List[List[float]]) -> List[List[float]]:
    cells: List[List[float]] = []
    for r in range(9):
        for c in range(9):
            value = grid[r][c]
            conf = confidence[r][c]
            if value != 0 and conf < UNCERTAIN_CLUE_THRESHOLD:
                cells.append([r, c, value, round(conf, 4)])
    return cells


def _prune_conflicting_digits(grid: List[List[int]], confidence: List[List[float]]) -> None:
    def dedupe_unit(coords: List[Tuple[int, int]]) -> bool:
        changed = False
        by_digit: Dict[int, List[Tuple[int, int]]] = {}
        for row, col in coords:
            value = grid[row][col]
            if value == 0:
                continue
            by_digit.setdefault(value, []).append((row, col))

        for dup_cells in by_digit.values():
            if len(dup_cells) <= 1:
                continue
            keep = max(dup_cells, key=lambda rc: confidence[rc[0]][rc[1]])
            for row, col in dup_cells:
                if (row, col) == keep:
                    continue
                grid[row][col] = 0
                confidence[row][col] = 0.0
                changed = True
        return changed

    def row_coords(row: int) -> List[Tuple[int, int]]:
        return [(row, col) for col in range(9)]

    def col_coords(col: int) -> List[Tuple[int, int]]:
        return [(row, col) for row in range(9)]

    def box_coords(box_r: int, box_c: int) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(box_r, box_r + 3) for c in range(box_c, box_c + 3)]

    changed = True
    while changed:
        changed = False
        for row in range(9):
            changed = dedupe_unit(row_coords(row)) or changed
        for col in range(9):
            changed = dedupe_unit(col_coords(col)) or changed
        for box_r in range(0, 9, 3):
            for box_c in range(0, 9, 3):
                coords = box_coords(box_r, box_c)
                changed = dedupe_unit(coords) or changed


def _drop_low_confidence_clues(grid: List[List[int]], confidence: List[List[float]], min_conf: float) -> int:
    dropped = 0
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                continue
            if confidence[r][c] < min_conf:
                grid[r][c] = 0
                confidence[r][c] = 0.0
                dropped += 1
    return dropped


def _is_consistent(grid: List[List[int]]) -> bool:
    for i in range(9):
        row_vals = [v for v in grid[i] if v != 0]
        col_vals = [grid[r][i] for r in range(9) if grid[r][i] != 0]
        if len(row_vals) != len(set(row_vals)):
            return False
        if len(col_vals) != len(set(col_vals)):
            return False

    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            vals = []
            for r in range(box_r, box_r + 3):
                for c in range(box_c, box_c + 3):
                    if grid[r][c] != 0:
                        vals.append(grid[r][c])
            if len(vals) != len(set(vals)):
                return False
    return True


def _render_debug_overlay(
    image: np.ndarray,
    boxes: List[CellBox],
    grid: List[List[int]],
    confidence: List[List[float]],
) -> np.ndarray:
    canvas = image.copy()
    for box in boxes:
        value = grid[box.row][box.col]
        conf = confidence[box.row][box.col]
        color = (0, 200, 0) if value != 0 else (120, 120, 120)
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), color, 1)
        if value != 0:
            label = f"{value}:{conf:.2f}"
            y = max(box.y1 + 16, 16)
            cv2.putText(
                canvas,
                label,
                (box.x1 + 3, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (10, 10, 220),
                1,
                cv2.LINE_AA,
            )
    return canvas


def _run_pipeline_internal(
    image: np.ndarray,
    with_debug_overlay: bool = False,
    ocr_mode: str = "tesseract",
) -> Tuple[Dict[str, object], Optional[np.ndarray]]:
    t_start = time.perf_counter()
    ocr_mode = "tesseract"

    t0 = time.perf_counter()
    boxes = detect_cells(image)
    t_detect = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    grid, confidence = _build_grid(image, boxes, ocr_mode=ocr_mode)
    t_ocr = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    _prune_conflicting_digits(grid, confidence)
    t_prune = (time.perf_counter() - t0) * 1000.0
    dropped_low_conf = _drop_low_confidence_clues(grid, confidence, min_conf=MIN_GIVEN_CONFIDENCE)

    result = _solve_from_grid(
        grid=grid,
        confidence=confidence,
        ocr_mode=ocr_mode,
        num_cells_detected=len(boxes),
    )
    needs_recovery = (
        not bool(result.get("solved", False))
        and int(result.get("solution_count_capped", 0)) >= 2
        and int(result.get("num_clues_detected", 0)) >= 17
    )
    if needs_recovery:
        recovered = _recover_missing_givens_for_ambiguity(image, boxes, grid, confidence)
        if recovered is not None:
            rec_grid, rec_conf, rec_count = recovered
            rec_result = _solve_from_grid(
                grid=rec_grid,
                confidence=rec_conf,
                ocr_mode=ocr_mode,
                num_cells_detected=len(boxes),
            )
            if bool(rec_result.get("solved", False)):
                rec_result["recovered_givens_count"] = rec_count
                rec_result["status"] = "ok"
                result = rec_result
    result["timing_ms"] = result.get("timing_ms", {})
    result["timing_ms"]["detect"] = round(t_detect, 2)
    result["timing_ms"]["ocr"] = round(t_ocr, 2)
    result["timing_ms"]["prune"] = round(t_prune, 2)
    result["num_low_conf_clues_dropped"] = dropped_low_conf

    t0 = time.perf_counter()
    debug_image = _render_debug_overlay(image, boxes, result["initial_grid"], confidence) if with_debug_overlay else None
    t_debug = (time.perf_counter() - t0) * 1000.0
    result["timing_ms"]["debug_overlay"] = round(t_debug, 2)
    result["timing_ms"]["total"] = round((time.perf_counter() - t_start) * 1000.0, 2)
    return result, debug_image


def _recover_missing_givens_for_ambiguity(
    image: np.ndarray,
    boxes: List[CellBox],
    grid: List[List[int]],
    confidence: List[List[float]],
) -> Optional[Tuple[List[List[int]], List[List[float]], int]]:
    box_by_rc = {(b.row, b.col): b for b in boxes}
    proposals: List[Tuple[int, int, int, float]] = []
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                continue
            box = box_by_rc.get((r, c))
            if box is None:
                continue
            cell = crop_cell(image, box)
            candidates = read_digit_candidates_tesseract_relaxed(cell, min_confidence=24.0, max_candidates=2)
            for digit, conf in candidates:
                proposals.append((r, c, digit, conf))

    if not proposals:
        return None

    proposals.sort(key=lambda item: item[3], reverse=True)
    proposals = proposals[:18]
    seen_rc = set()
    top_single: List[Tuple[int, int, int, float]] = []
    for p in proposals:
        rc = (p[0], p[1])
        if rc in seen_rc:
            continue
        seen_rc.add(rc)
        top_single.append(p)

    def is_consistent_plus(base: List[List[int]], r: int, c: int, value: int) -> bool:
        if any(base[r][x] == value for x in range(9)):
            return False
        if any(base[y][c] == value for y in range(9)):
            return False
        br = (r // 3) * 3
        bc = (c // 3) * 3
        for y in range(br, br + 3):
            for x in range(bc, bc + 3):
                if base[y][x] == value:
                    return False
        return True

    def try_grid_with_added(added: List[Tuple[int, int, int, float]]) -> Optional[Tuple[List[List[int]], List[List[float]], int]]:
        g = [row[:] for row in grid]
        conf = [row[:] for row in confidence]
        for r, c, digit, score in added:
            if g[r][c] != 0:
                continue
            if not is_consistent_plus(g, r, c, digit):
                return None
            g[r][c] = digit
            conf[r][c] = score
        cnt, exhausted = count_solutions_with_budget([row[:] for row in g], limit=2, max_nodes=300_000)
        if exhausted or cnt != 1:
            return None
        solved_grid = [row[:] for row in g]
        solved_ok, solved_exhausted = solve_sudoku_with_budget(solved_grid, max_nodes=2_000_000)
        if solved_exhausted or not solved_ok:
            return None
        return g, conf, len(added)

    for p in top_single:
        tried = try_grid_with_added([p])
        if tried is not None:
            return tried

    pair_pool = top_single[:10]
    for i in range(len(pair_pool)):
        for j in range(i + 1, len(pair_pool)):
            a = pair_pool[i]
            b = pair_pool[j]
            if (a[0], a[1]) == (b[0], b[1]):
                continue
            tried = try_grid_with_added([a, b])
            if tried is not None:
                return tried
    return None


def _solve_from_grid(
    grid: List[List[int]],
    confidence: List[List[float]],
    ocr_mode: str,
    num_cells_detected: int,
) -> Dict[str, object]:
    initial = [row[:] for row in grid]
    clues = _count_clues(initial)
    grid_is_consistent = _is_consistent(initial)
    uncertainty_metrics = _uncertainty_stats(initial, confidence)
    solve_eligible = clues >= 17 and grid_is_consistent
    is_solved = False
    solved = [row[:] for row in initial]
    clues_used = [row[:] for row in initial]
    removed_for_relaxation = 0
    solution_count_capped = 0
    t0 = time.perf_counter()
    if solve_eligible:
        count_grid = [row[:] for row in initial]
        solution_count_capped, count_exhausted = count_solutions_with_budget(
            count_grid, limit=2, max_nodes=200_000
        )
        if not count_exhausted and solution_count_capped != 1:
            is_solved = False
        else:
            if count_exhausted:
                solution_count_capped = 1
            solved_ok, _solve_exhausted = solve_sudoku_with_budget(solved, max_nodes=2_000_000)
            is_solved = bool(solved_ok)
            if is_solved:
                clues_used = [row[:] for row in initial]
                solution_count_capped = max(1, solution_count_capped)
    t_solve = (time.perf_counter() - t0) * 1000.0

    actions: List[Tuple[int, int, int]] = []
    if is_solved:
        actions = _actions_from_solution(initial, solved)

    if not grid_is_consistent:
        status = "invalid_grid_from_detection_or_ocr"
    elif clues < 17:
        status = "insufficient_clues_for_reliable_solve"
    elif is_solved:
        status = "ok"
    else:
        status = "unsolved"

    confidence_metrics = _confidence_stats(initial, confidence)
    result = {
        "status": status,
        "solved": is_solved,
        "initial_grid": initial,
        "confidence_grid": confidence,
        "solved_grid": solved if is_solved else None,
        "actions": actions,
        "num_cells_detected": num_cells_detected,
        "num_clues_detected": clues,
        "num_clues_used_for_solve": _count_clues(clues_used),
        "num_removed_clues_for_relaxation": removed_for_relaxation,
        "solution_count_capped": solution_count_capped,
        "solve_eligible": solve_eligible,
        "ocr_mode": ocr_mode,
        "uncertain_cells": _uncertain_cells(initial, confidence),
        **confidence_metrics,
        **uncertainty_metrics,
    }
    result["timing_ms"] = {
        "solve": round(t_solve, 2),
    }
    return result


def run_pipeline(image: np.ndarray, ocr_mode: str = "tesseract") -> Dict[str, object]:
    result, _ = _run_pipeline_internal(image, with_debug_overlay=False, ocr_mode=ocr_mode)
    return result


def run_pipeline_debug(image: np.ndarray, ocr_mode: str = "tesseract") -> Tuple[Dict[str, object], np.ndarray]:
    result, debug_image = _run_pipeline_internal(image, with_debug_overlay=True, ocr_mode=ocr_mode)
    if debug_image is None:
        debug_image = image.copy()
    return result, debug_image


def run_pipeline_from_grid(
    initial_grid: List[List[int]],
    confidence_grid: List[List[float]],
    ocr_mode: str = "tesseract",
) -> Dict[str, object]:
    return _solve_from_grid(
        grid=[row[:] for row in initial_grid],
        confidence=[row[:] for row in confidence_grid],
        ocr_mode=ocr_mode,
        num_cells_detected=81,
    )
