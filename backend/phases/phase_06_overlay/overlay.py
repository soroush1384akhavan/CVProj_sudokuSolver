from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from common.images import save_image

Board = list[list[int]]


def draw_solution_on_warped(warped_bgr: np.ndarray, original_board: Board, solved_board: Board) -> np.ndarray:
    out = warped_bgr.copy()
    h, w = out.shape[:2]
    cell_h = h // 9
    cell_w = w // 9
    for r in range(9):
        for c in range(9):
            if original_board[r][c] == 0 and solved_board[r][c] != 0:
                text = str(solved_board[r][c])
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.15
                thickness = 2
                size, _ = cv2.getTextSize(text, font, scale, thickness)
                x = c * cell_w + (cell_w - size[0]) // 2
                y = r * cell_h + (cell_h + size[1]) // 2
                cv2.putText(out, text, (x, y), font, scale, (20, 90, 255), thickness, cv2.LINE_AA)
    return out


def draw_solution_overlay_only(board_size: int, original_board: Board, solved_board: Board) -> tuple[np.ndarray, np.ndarray]:

    overlay = np.zeros((board_size, board_size, 3), dtype=np.uint8)
    mask = np.zeros((board_size, board_size), dtype=np.uint8)

    cell = board_size // 9
    for r in range(9):
        for c in range(9):
            if original_board[r][c] == 0 and solved_board[r][c] != 0:
                text = str(solved_board[r][c])
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.15
                thickness = 2
                size, _ = cv2.getTextSize(text, font, scale, thickness)
                x = c * cell + (cell - size[0]) // 2
                y = r * cell + (cell + size[1]) // 2
                cv2.putText(overlay, text, (x, y), font, scale, (20, 90, 255), thickness, cv2.LINE_AA)
                cv2.putText(mask, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)

    return overlay, mask


def create_overlay_images(
    run_dir: Path,
    original_bgr: np.ndarray,
    warped_bgr: np.ndarray,
    corners: np.ndarray,
    original_board: Board,
    solved_board: Board,
    inverse_matrix: np.ndarray | None = None,
) -> list[dict[str, str]]:
    board_size = warped_bgr.shape[0]

    solved_warped = draw_solution_on_warped(warped_bgr, original_board, solved_board)
    save_image(run_dir / "08_solved_warped.png", solved_warped)

    if inverse_matrix is None:
        h, w = original_bgr.shape[:2]
        src = np.array(
            [[0, 0], [board_size - 1, 0], [board_size - 1, board_size - 1], [0, board_size - 1]],
            dtype="float32",
        )
        inverse_matrix = cv2.getPerspectiveTransform(src, corners.astype("float32"))

    h, w = original_bgr.shape[:2]

    digit_overlay, digit_mask = draw_solution_overlay_only(board_size, original_board, solved_board)
    projected_overlay = cv2.warpPerspective(digit_overlay, inverse_matrix, (w, h))
    projected_mask = cv2.warpPerspective(digit_mask, inverse_matrix, (w, h))

    result = original_bgr.copy()
    mask_bool = projected_mask > 0
    result[mask_bool] = projected_overlay[mask_bool]

    save_image(run_dir / "09_solved_original_overlay.png", result)

    return [
        {
            "key": "solved_warped",
            "title": "Solved Warped Board",
            "description": "Solved digits drawn on the straightened Sudoku board.",
            "filename": "08_solved_warped.png",
        },
        {
            "key": "solved_overlay",
            "title": "Solved Original Overlay",
            "description": "Solved board projected back onto the original image.",
            "filename": "09_solved_original_overlay.png",
        },
    ]