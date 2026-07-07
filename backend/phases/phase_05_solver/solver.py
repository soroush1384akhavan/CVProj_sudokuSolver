from __future__ import annotations

from copy import deepcopy
from phases.phase_05_solver.validator import validate_board_rules, Board


def candidates_for(board: Board, row: int, col: int) -> set[int]:
    if board[row][col] != 0:
        return set()
    used = set(board[row])
    used.update(board[r][col] for r in range(9))
    box_r = (row // 3) * 3
    box_c = (col // 3) * 3
    used.update(board[r][c] for r in range(box_r, box_r + 3) for c in range(box_c, box_c + 3))
    return {n for n in range(1, 10) if n not in used}

# MRV : find cell with least candidate
def find_best_empty_cell(board: Board) -> tuple[int, int, set[int]] | None:
    best: tuple[int, int, set[int]] | None = None
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                cand = candidates_for(board, r, c)
                if not cand:
                    return (r, c, set())
                if best is None or len(cand) < len(best[2]):
                    best = (r, c, cand)
    return best


def solve_in_place(board: Board) -> bool:
    empty = find_best_empty_cell(board)
    if empty is None:
        return True
    row, col, cand = empty
    if not cand:
        return False
    for num in sorted(cand):
        board[row][col] = num
        if solve_in_place(board):
            return True
        board[row][col] = 0
    return False


def solve_board(board: Board) -> tuple[bool, Board | None, str]:
    valid, message = validate_board_rules(board)
    if not valid:
        return False, None, message
    working = deepcopy(board) # for copy all dimantions
    solved = solve_in_place(working)
    if not solved:
        return False, None, "This puzzle has no valid solution."
    return True, working, "Puzzle solved successfully."
