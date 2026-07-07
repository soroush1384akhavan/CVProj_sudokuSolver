from __future__ import annotations

Board = list[list[int]]


def validate_shape(board: Board) -> None:
    if len(board) != 9 or any(len(row) != 9 for row in board):
        raise ValueError("Board must be a 9x9 matrix.")
    for row in board:
        for value in row:
            if not isinstance(value, int) or value < 0 or value > 9:
                raise ValueError("Board values must be integers from 0 to 9.")


def has_duplicates(values: list[int]) -> bool:
    non_zero = [v for v in values if v != 0]
    return len(non_zero) != len(set(non_zero))


def validate_board_rules(board: Board) -> tuple[bool, str]:
    try:
        validate_shape(board)
    except ValueError as exc:
        return False, str(exc)

    for r in range(9):
        if has_duplicates(board[r]):
            return False, f"Duplicate digit in row {r + 1}."

    for c in range(9):
        column = [board[r][c] for r in range(9)]
        if has_duplicates(column):
            return False, f"Duplicate digit in column {c + 1}."

    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            values = [board[r][c] for r in range(box_r, box_r + 3) for c in range(box_c, box_c + 3)]
            if has_duplicates(values):
                return False, f"Duplicate digit in 3x3 box starting at row {box_r + 1}, col {box_c + 1}."

    return True, "Board is valid."
