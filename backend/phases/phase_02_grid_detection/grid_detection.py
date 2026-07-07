from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from common.images import save_image

logger = logging.getLogger(__name__)


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = points.reshape(4, 2).astype("float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left

    return ordered


def fallback_corners(image: np.ndarray) -> np.ndarray:
    """Use full image bounds as fallback corners."""
    h, w = image.shape[:2]
    return np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype="float32",
    )


def _ensure_binary_image(image: np.ndarray) -> np.ndarray:
    """
    Ensure input is a single-channel binary uint8 image with values 0/255.

    Expected foreground:
        white grid/lines on black background.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    return binary


def _quad_side_lengths(corners: np.ndarray) -> np.ndarray:
    """Return the 4 side lengths of an ordered quadrilateral: TL, TR, BR, BL."""
    rolled = np.roll(corners, -1, axis=0)
    return np.linalg.norm(rolled - corners, axis=1)


def _is_plausible_square(
    corners: np.ndarray,
    min_side: float = 20.0,
    max_aspect_ratio: float = 1.4,
) -> bool:
    sides = _quad_side_lengths(corners)

    if np.any(sides < min_side):
        return False

    if sides.max() / sides.min() > max_aspect_ratio:
        return False

    return True


def _find_quad_candidate(
    binary: np.ndarray,
    max_contours: int,
    min_area: float,
    epsilon_ratios: list[float],
) -> np.ndarray | None:
    """Find the best quadrilateral candidate from a binary preprocessed image."""

    contours, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        peri = cv2.arcLength(contour, True)

        for epsilon_ratio in epsilon_ratios:
            approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)

            if len(approx) != 4:
                continue

            if not cv2.isContourConvex(approx):
                continue

            ordered = order_points(approx)

            if not _is_plausible_square(ordered):
                continue

            return ordered

    return None


def _repair_grid_lines(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Repair broken grid lines in the binary image."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )

    repaired = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    return repaired


def _draw_detected_corners(binary: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Create BGR debug image from binary image and draw detected contour/corners."""
    debug_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    cv2.polylines(
        debug_bgr,
        [corners.astype("int32")],
        isClosed=True,
        color=(0, 255, 0),
        thickness=5,
    )

    for idx, point in enumerate(corners.astype("int32")):
        cv2.circle(debug_bgr, tuple(point), 10, (0, 0, 255), -1)
        cv2.putText(
            debug_bgr,
            str(idx + 1),
            tuple(point + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    return debug_bgr


def find_sudoku_grid(
    preprocessed_binary: np.ndarray,
    output_dir: Path,
) -> dict[str, np.ndarray | bool | str]:
    """
    Detect Sudoku grid from a preprocessed binary image.

    Input:
        preprocessed_binary:
            Single-channel binary image from phase 01 preprocessing.
            Expected format: uint8, 0/255, grid/foreground white.

    Output:
        warped_binary:
            Perspective-corrected binary board.

        warped:
            BGR-compatible version of warped_binary for old downstream code
            that still expects warped_bgr.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    binary = _ensure_binary_image(preprocessed_binary)

    max_contours = int(settings.get("grid_detection.max_contours_to_check", 20))
    min_area = float(settings.get("grid_detection.min_contour_area", 10000))
    base_epsilon_ratio = float(settings.get("grid_detection.approx_epsilon_ratio", 0.02))
    board_size = int(settings.get("grid_detection.board_size", 450))

    epsilon_ratios = sorted(
        {
            base_epsilon_ratio,
            base_epsilon_ratio * 1.5,
            base_epsilon_ratio * 2.5,
        }
    )

    corners = _find_quad_candidate(
        binary=binary,
        max_contours=max_contours,
        min_area=min_area,
        epsilon_ratios=epsilon_ratios,
    )

    if corners is None:
        logger.warning("No clean quad contour found, retrying after morphological repair.")

        repaired = _repair_grid_lines(binary)

        corners = _find_quad_candidate(
            binary=repaired,
            max_contours=max_contours,
            min_area=min_area,
            epsilon_ratios=epsilon_ratios,
        )

    found = corners is not None

    if not found:
        logger.warning("Grid detection failed, falling back to full binary image bounds.")
        corners = fallback_corners(binary)

    dst = np.array(
        [
            [0, 0],
            [board_size - 1, 0],
            [board_size - 1, board_size - 1],
            [0, board_size - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(corners, dst)

    det = np.linalg.det(matrix)

    if abs(det) < 1e-6:
        logger.warning("Degenerate perspective matrix, falling back to full binary image bounds.")

        found = False
        corners = fallback_corners(binary)
        matrix = cv2.getPerspectiveTransform(corners, dst)

    warped_binary = cv2.warpPerspective(
        binary,
        matrix,
        (board_size, board_size),
        flags=cv2.INTER_NEAREST,
    )

    _, warped_binary = cv2.threshold(
        warped_binary,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    inverse_matrix = np.linalg.inv(matrix)

    contour_debug = _draw_detected_corners(binary, corners)

    # برای سازگاری با کدهای قبلی که warped_bgr می‌خواستند
    warped_bgr = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR)

    save_image(output_dir / "05_detected_contour.png", contour_debug)
    save_image(output_dir / "06_warped_board_binary.png", warped_binary)
    save_image(output_dir / "06_warped_board.png", warped_bgr)

    return {
        "found": found,
        "corners": corners,
        "matrix": matrix,
        "inverse_matrix": inverse_matrix,

        # خروجی اصلی درست
        "warped_binary": warped_binary,

        # برای compatibility با فاز بعدی اگر هنوز اسمش warped_bgr است
        "warped": warped_bgr,

        "contour_debug_path": "05_detected_contour.png",
        "warped_binary_path": "06_warped_board_binary.png",
        "warped_path": "06_warped_board.png",
    }