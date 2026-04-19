from __future__ import annotations

from typing import List, Optional, Tuple

Grid = List[List[int]]


def _find_empty_with_mrv(grid: Grid) -> Optional[Tuple[int, int]]:
    best_cell: Optional[Tuple[int, int]] = None
    best_count = 10
    for row in range(9):
        for col in range(9):
            if grid[row][col] != 0:
                continue
            count = len(_candidates(grid, row, col))
            if count < best_count:
                best_count = count
                best_cell = (row, col)
            if best_count == 1:
                return best_cell
    return best_cell


def _candidates(grid: Grid, row: int, col: int) -> List[int]:
    used = set()
    used.update(grid[row][x] for x in range(9))
    used.update(grid[y][col] for y in range(9))

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for y in range(box_row, box_row + 3):
        for x in range(box_col, box_col + 3):
            used.add(grid[y][x])

    return [value for value in range(1, 10) if value not in used]


def _is_valid_assignment(grid: Grid, row: int, col: int, value: int) -> bool:
    if any(grid[row][x] == value for x in range(9)):
        return False
    if any(grid[y][col] == value for y in range(9)):
        return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for y in range(box_row, box_row + 3):
        for x in range(box_col, box_col + 3):
            if grid[y][x] == value:
                return False
    return True


def solve_sudoku_with_budget(grid: Grid, max_nodes: int) -> Tuple[bool, bool]:
    budget = [max_nodes]

    def _solve() -> Tuple[bool, bool]:
        budget[0] -= 1
        if budget[0] < 0:
            return False, True

        next_cell = _find_empty_with_mrv(grid)
        if next_cell is None:
            return True, False

        row, col = next_cell
        for value in _candidates(grid, row, col):
            if not _is_valid_assignment(grid, row, col, value):
                continue
            grid[row][col] = value
            solved, exhausted = _solve()
            if exhausted:
                grid[row][col] = 0
                return False, True
            if solved:
                return True, False
            grid[row][col] = 0
        return False, False

    return _solve()


def count_solutions_with_budget(grid: Grid, limit: int, max_nodes: int) -> Tuple[int, bool]:
    budget = [max_nodes]

    def _count(local_limit: int) -> Tuple[int, bool]:
        budget[0] -= 1
        if budget[0] < 0:
            return 0, True

        next_cell = _find_empty_with_mrv(grid)
        if next_cell is None:
            return 1, False

        row, col = next_cell
        total = 0
        for value in _candidates(grid, row, col):
            if not _is_valid_assignment(grid, row, col, value):
                continue
            grid[row][col] = value
            found, exhausted = _count(local_limit - total)
            grid[row][col] = 0
            if exhausted:
                return total, True
            total += found
            if total >= local_limit:
                return local_limit, False
        return total, False

    return _count(limit)
