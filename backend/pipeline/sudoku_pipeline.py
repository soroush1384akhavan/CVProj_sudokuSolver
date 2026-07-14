from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from common.file_store import create_run_dir, save_json, load_json
from common.images import imread_color, save_image
from common.paths import public_url_for_run_file, run_dir
from app.config import settings
from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid
from phases.phase_03_cell_extraction.cell_extraction import extract_cells
from phases.phase_04_digit_recognition.classifier import PyTorchDigitClassifier
from phases.phase_05_solver.solver import solve_board
from phases.phase_06_overlay.overlay import create_overlay_images

Board = list[list[int]]


def phase_item(run_id: str, key: str, title: str, description: str, filename: str | None) -> dict[str, str | None]:
    return {
        "key": key,
        "title": title,
        "description": description,
        "image_url": public_url_for_run_file(run_id, filename) if filename else None,
    }


def run_prediction_pipeline(image_path: Path, language: str = "en") -> dict[str, Any]:
    run_id, out_dir = create_run_dir()
    image_bgr = imread_color(image_path)
    save_image(out_dir / "00_uploaded.png", image_bgr)

    phase1 = preprocess_image(image_bgr, out_dir)
    phase2 = find_sudoku_grid(phase1["original"], phase1["threshold"], out_dir)  # type: ignore[arg-type]
    phase3 = extract_cells(phase2["warped"], phase2["warped_binary"] ,out_dir)  # type: ignore[arg-type]

    print(f"[PIPELINE] selected language: {language}")
    classifier = classifier = PyTorchDigitClassifier(language=language)
    board, confidence, low_confidence_cells = classifier.predict_board(
        phase3["clean_cells"],  # type: ignore[arg-type]
        phase3["empty_flags"],  # type: ignore[arg-type]
    )

    metadata = {
        "run_id": run_id,
        "corners": np.asarray(phase2["corners"]).tolist(),
        "grid_found": bool(phase2["found"]),
        "board": board,
        "confidence": confidence,
        "model_status": classifier.status_message,
    }
    save_json(out_dir / "metadata.json", metadata)

    phases = [
        phase_item(run_id, "original", "Original Image", "The uploaded Sudoku image.", "00_uploaded.png"),
        phase_item(run_id, "grayscale", "Phase 1 — Grayscale", "Input converted to grayscale.", "02_grayscale.png"),
        phase_item(run_id, "blur", "Phase 1 — Blur", "Noise-reduced grayscale image.", "03_blur.png"),
        phase_item(run_id, "threshold", "Phase 1 — Threshold", "Adaptive threshold image used for contour detection.", "04_threshold.png"),
        phase_item(run_id, "contour", "Phase 2 — Detected Grid", "Largest four-corner grid candidate.", "05_detected_contour.png"),
        phase_item(run_id, "warped", "Phase 2 — Warped Board", "Perspective-corrected Sudoku board.", "06_warped_board.png"),
    ]

    if bool(settings.get("ui.show_cells_separately", True)):
        for idx, filename in enumerate(phase3.get("cell_filenames", [])):  # type: ignore[union-attr]
            row, col = divmod(idx, 9)
            phases.append(
                phase_item(
                    run_id,
                    f"cell_{idx:02d}",
                    f"Phase 3 — Cell {idx:02d}",
                    f"Separate extracted cell at row {row + 1}, column {col + 1}.",
                    filename,
                )
            )
    else:
        phases.append(phase_item(run_id, "cells", "Phase 3 — Extracted Cells", "Montage of all 81 cleaned cells.", "07_cells_montage.png"))

    message = "Image processed. Review the board and correct cells if needed."
    if not bool(phase2["found"]):
        message += " Grid contour was not confidently found; full image fallback was used."

    return {
        "success": True,
        "run_id": run_id,
        "board": board,
        "confidence": confidence,
        "low_confidence_cells": low_confidence_cells,
        "phases": phases,
        "message": message,
        "model_status": classifier.status_message,
    }


def run_solve_pipeline(board: Board, run_id: str | None, original_board: Board | None = None) -> dict[str, Any]:
    solved, solved_board, message = solve_board(board)
    phases: list[dict[str, str | None]] = []

    if solved and solved_board and run_id:
        rdir = run_dir(run_id)
        metadata_path = rdir / "metadata.json"
        original_path = rdir / "00_uploaded.png"
        warped_path = rdir / "06_warped_board.png"
        if metadata_path.exists() and original_path.exists() and warped_path.exists():
            metadata = load_json(metadata_path)
            corners = np.array(metadata["corners"], dtype="float32")
            original_bgr = imread_color(original_path)
            warped_bgr = imread_color(warped_path)
            base_board = original_board or board
            overlay_items = create_overlay_images(rdir, original_bgr, warped_bgr, corners, base_board, solved_board)
            for item in overlay_items:
                phases.append(phase_item(run_id, item["key"], item["title"], item["description"], item["filename"]))

    return {
        "success": solved,
        "solved_board": solved_board,
        "phases": phases,
        "message": message,
    }
