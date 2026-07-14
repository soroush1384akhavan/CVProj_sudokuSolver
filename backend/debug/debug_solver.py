## for run : python -m debug.debug_solver

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid
from phases.phase_03_cell_extraction.cell_extraction import extract_cells
from phases.phase_04_digit_recognition.src.classifier import PyTorchDigitClassifier
from phases.phase_05_solver.solver import solve_board

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DATASETS = [
    {
        "name": "v1",
        "folder": "storage/sudoku/raw/v1_test/v1_test",
        "exclude_dirs": {"mixed", "mixed 2"},
    },
    {
        "name": "v2",
        "folder": "storage/sudoku/raw/v2_test/v2_test",
        "exclude_dirs": {"mixed", "mixed 2"},
    },
]


# ---------------------------------------------------------------------------
# Optional ground-truth loading (برای محاسبه دقت واقعی تشخیص ارقام، اختیاری)
# ---------------------------------------------------------------------------
def _load_ground_truth(image_path: Path) -> Optional[list[list[int]]]:

    json_path = image_path.with_suffix(".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            board = data["board"] if isinstance(data, dict) and "board" in data else data
            return [[int(v) for v in row] for row in board]
        except Exception:
            return None

    txt_path = image_path.with_suffix(".txt")
    if txt_path.exists():
        try:
            lines = [
                line.strip()
                for line in txt_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            board = []
            for line in lines[:9]:
                row = [0 if ch in "0." else int(ch) for ch in line[:9]]
                board.append(row)
            if len(board) == 9 and all(len(r) == 9 for r in board):
                return board
        except Exception:
            return None

    return None


def _board_accuracy(pred: list[list[int]], truth: list[list[int]]) -> float:
    """Fraction of correctly recognized non-empty ground-truth cells."""
    total = sum(1 for r in range(9) for c in range(9) if truth[r][c] != 0)
    if total == 0:
        return float("nan")
    correct = sum(
        1
        for r in range(9)
        for c in range(9)
        if truth[r][c] != 0 and pred[r][c] == truth[r][c]
    )
    return correct / total


def _safe_mean(values: list) -> float:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


# ---------------------------------------------------------------------------
# Core pipeline (per dataset)
# ---------------------------------------------------------------------------
def load_and_solve(
    folder: str | Path,
    debug_output_dir: str | Path,
    exclude_dirs: set[str] | None = None,
    dataset_name: str = "default",
) -> list:

    folder = Path(folder)
    debug_output_dir = Path(debug_output_dir)
    exclude_dirs = exclude_dirs or {"mixed", "mixed 2"}

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted(
        p
        for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(excluded in p.parts for excluded in exclude_dirs)
    )

    if not image_paths:
        raise ValueError(f"No images found in: {folder}")

    classifier = PyTorchDigitClassifier()

    results = []
    solve_failed_count = 0
    grid_not_found_count = 0

    for path in image_paths:
        start_time = time.perf_counter()

        image_bgr = cv2.imread(str(path))

        if image_bgr is None:
            print(f"⚠️  Failed to read image (skipped): {path}")
            continue

        image_output_dir = debug_output_dir / path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        preprocessed = preprocess_image(image_bgr, image_output_dir)

        grid_result = find_sudoku_grid(
            preprocessed_binary=preprocessed["threshold"],
            original_bgr=preprocessed["original"],
            output_dir=image_output_dir,
        )

        if not grid_result["found"]:
            grid_not_found_count += 1
            print(f"⚠️  Grid not found (fallback used): {path.name}")

        cells_result = extract_cells(
            warped_bgr=grid_result["warped"],
            warped_binary=grid_result["warped_binary"],
            output_dir=image_output_dir,
        )

        original_board, confidence, low_confidence_cells = classifier.predict_board(
            clean_cells=cells_result["clean_cells"],
            empty_flags=cells_result["empty_flags"],
        )

        solved, solved_board, message = solve_board(original_board)

        if not solved:
            solve_failed_count += 1
            print(f"⚠️  Solve failed: {path.name} -> {message}")

        elapsed = time.perf_counter() - start_time

        ground_truth = _load_ground_truth(path)
        digit_accuracy = (
            _board_accuracy(original_board, ground_truth) if ground_truth is not None else None
        )

        # Failure category used later for error analysis / reporting.
        if not grid_result["found"]:
            status = "grid_not_found"
        elif not solved:
            status = "solve_failed"
        elif len(low_confidence_cells) > 0:
            status = "low_confidence"
        else:
            status = "success"

        results.append(
            {
                "dataset": dataset_name,
                "source_path": str(path),
                "original_board": original_board,
                "solved_board": solved_board,
                "solved": solved,
                "message": message,
                "confidence": confidence,
                "low_confidence_cells": low_confidence_cells,
                "grid_found": grid_result["found"],
                "elapsed_seconds": elapsed,
                "ground_truth": ground_truth,
                "digit_accuracy": digit_accuracy,
                "status": status,
            }
        )

    print(
        f"\n[{dataset_name}] Total images: {len(results)} | "
        f"Grid not found: {grid_not_found_count} | "
        f"Solve failed: {solve_failed_count}"
    )

    return results


def load_and_solve_all(datasets: list[dict], debug_output_dir: str | Path) -> list:
    """Run load_and_solve for every configured dataset version (v1, v2, ...)."""
    all_results = []
    for ds in datasets:
        print(f"\n{'=' * 60}\nProcessing dataset: {ds['name']}  ({ds['folder']})\n{'=' * 60}")
        folder = Path(ds["folder"])
        if not folder.is_dir():
            print(f"⚠️  Skipping dataset '{ds['name']}': folder not found -> {folder}")
            continue

        ds_output_dir = Path(debug_output_dir) / ds["name"]
        results = load_and_solve(
            folder=folder,
            debug_output_dir=ds_output_dir,
            exclude_dirs=ds.get("exclude_dirs"),
            dataset_name=ds["name"],
        )
        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# Console debug printing
# ---------------------------------------------------------------------------
def print_board(board: list[list[int]], title: str = "Board") -> None:
    print(f"\n{title}")

    for r in range(9):
        row_str = " ".join(str(board[r][c]) if board[r][c] != 0 else "." for c in range(9))

        if r % 3 == 0 and r != 0:
            print("-" * 21)

        formatted = " | ".join(row_str[i : i + 6] for i in range(0, len(row_str), 6))
        print(formatted)


def debug_print_boards(results, num_samples: int = 5, seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(results))
    random_indices = random.sample(range(len(results)), k=num_samples)

    for idx in random_indices:
        item = results[idx]

        print(f"\n{'=' * 40}")
        print(f"Image: {item['source_path']}  (dataset={item['dataset']})")
        print(
            f"Grid found: {item['grid_found']} | "
            f"Solved: {item['solved']} | "
            f"Status: {item['status']} | "
            f"Message: {item['message']}"
        )

        print_board(item["original_board"], title="Detected original board")

        if item["solved_board"] is not None:
            print_board(item["solved_board"], title="Solved board")

        if item["low_confidence_cells"]:
            print(f"Low-confidence cells: {item['low_confidence_cells']}")

        if item["digit_accuracy"] is not None:
            print(f"Digit accuracy (vs ground truth): {item['digit_accuracy']:.1%}")


def debug_show_confidence_heatmap(
    results,
    num_samples: int = 4,
    seed: int | None = None,
    title: str = "Digit confidence heatmaps",
) -> None:
    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(results))
    random_indices = random.sample(range(len(results)), k=num_samples)

    cols = 2
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, idx in enumerate(random_indices):
        item = results[idx]
        confidence = item["confidence"]

        plt.subplot(rows, cols, i + 1)
        im = plt.imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)

        for r in range(9):
            for c in range(9):
                digit = item["original_board"][r][c]
                if digit != 0:
                    plt.text(c, r, str(digit), ha="center", va="center", fontsize=9)

        plt.title(f"[{item['dataset']}] idx {idx} | solved={item['solved']}", fontsize=9)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Final comprehensive report (دقت / زمان اجرا / تحلیل خطا / گزارش نهایی)
# ---------------------------------------------------------------------------
def generate_final_report(results: list, report_path: str | Path) -> dict:

    report_path = Path(report_path)
    dataset_names = sorted(set(r["dataset"] for r in results))

    summary = {}
    lines = ["=" * 70, "SUDOKU SOLVER — FINAL PERFORMANCE REPORT", "=" * 70]

    for name in dataset_names + (["ALL"] if len(dataset_names) > 1 else []):
        subset = results if name == "ALL" else [r for r in results if r["dataset"] == name]
        if not subset:
            continue

        n = len(subset)
        grid_found_rate = sum(r["grid_found"] for r in subset) / n
        solve_rate = sum(r["solved"] for r in subset) / n
        avg_confidence = _safe_mean(
            [r["confidence"].mean() if hasattr(r["confidence"], "mean") else None for r in subset]
        )
        avg_low_conf_cells = _safe_mean([len(r["low_confidence_cells"]) for r in subset])
        avg_time = _safe_mean([r["elapsed_seconds"] for r in subset])
        total_time = sum(r["elapsed_seconds"] for r in subset)

        digit_accuracies = [r["digit_accuracy"] for r in subset if r["digit_accuracy"] is not None]
        avg_digit_accuracy = _safe_mean(digit_accuracies) if digit_accuracies else None

        status_counts: dict[str, int] = {}
        for r in subset:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

        failures = [r for r in subset if r["status"] != "success"]
        failure_samples = [
            {"path": r["source_path"], "status": r["status"], "message": r["message"]}
            for r in failures[:10]
        ]

        summary[name] = {
            "count": n,
            "grid_found_rate": grid_found_rate,
            "solve_rate": solve_rate,
            "avg_confidence": avg_confidence,
            "avg_low_confidence_cells": avg_low_conf_cells,
            "avg_time_seconds": avg_time,
            "total_time_seconds": total_time,
            "avg_digit_accuracy": avg_digit_accuracy,
            "status_counts": status_counts,
            "failure_samples": failure_samples,
        }

        lines.append(f"\n--- Dataset: {name} ---")
        lines.append(f"Images processed          : {n}")
        lines.append(f"Grid detection success    : {grid_found_rate:.1%}")
        lines.append(f"Solve success rate        : {solve_rate:.1%}")
        lines.append(
            f"Avg digit confidence      : {avg_confidence:.3f}"
            if not np.isnan(avg_confidence)
            else "Avg digit confidence      : N/A"
        )
        lines.append(f"Avg low-confidence cells  : {avg_low_conf_cells:.2f} / image")
        if avg_digit_accuracy is not None:
            lines.append(
                f"Digit recognition accuracy: {avg_digit_accuracy:.1%}  "
                f"(based on {len(digit_accuracies)} labeled images)"
            )
        else:
            lines.append("Digit recognition accuracy: N/A (no ground-truth files found)")
        lines.append(f"Avg time / image          : {avg_time:.3f}s")
        lines.append(f"Total time                : {total_time:.1f}s")
        lines.append(f"Status breakdown          : {status_counts}")

        if failure_samples:
            lines.append("Sample failures:")
            for f in failure_samples:
                lines.append(f"   - [{f['status']}] {f['path']} -> {f['message']}")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n📄 Full report saved to: {report_path}")

    return summary


def debug_show_summary_charts(results: list, summary: dict, output_dir: str | Path) -> None:
    """Bar/hist charts for accuracy, timing and error breakdown across datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [k for k in summary.keys() if k != "ALL"] or ["ALL"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    x = np.arange(len(dataset_names))
    width = 0.35

    # 1) Grid detection vs solve success rate
    ax = axes[0, 0]
    ax.bar(x - width / 2, [summary[d]["grid_found_rate"] for d in dataset_names], width, label="Grid found")
    ax.bar(x + width / 2, [summary[d]["solve_rate"] for d in dataset_names], width, label="Solved")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylim(0, 1.05)
    ax.set_title("System performance: grid detection & solve rate")
    ax.legend()

    # 2) Execution time distribution
    ax = axes[0, 1]
    for d in dataset_names:
        times = [r["elapsed_seconds"] for r in results if r["dataset"] == d]
        if times:
            ax.hist(times, bins=15, alpha=0.6, label=d)
    ax.set_title("Execution time distribution (s / image)")
    ax.set_xlabel("seconds")
    ax.legend()

    # 3) Error breakdown (stacked)
    ax = axes[1, 0]
    all_statuses = sorted(set(r["status"] for r in results))
    bottom = np.zeros(len(dataset_names))
    for status in all_statuses:
        counts = [
            sum(1 for r in results if r["dataset"] == d and r["status"] == status)
            for d in dataset_names
        ]
        ax.bar(dataset_names, counts, bottom=bottom, label=status)
        bottom += np.array(counts)
    ax.set_title("Error analysis: status breakdown")
    ax.legend()

    # 4) Confidence / digit accuracy
    ax = axes[1, 1]
    conf_means = [summary[d]["avg_confidence"] for d in dataset_names]
    acc_vals = [
        summary[d]["avg_digit_accuracy"] if summary[d]["avg_digit_accuracy"] is not None else 0
        for d in dataset_names
    ]
    ax.bar(x - width / 2, conf_means, width, label="Avg confidence")
    ax.bar(x + width / 2, acc_vals, width, label="Digit accuracy (if labeled)")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Digit recognition quality")
    ax.legend()

    plt.suptitle("Final sudoku system performance report", fontsize=14)
    plt.tight_layout()

    fig_path = output_dir / "final_summary_charts.png"
    plt.savefig(fig_path, dpi=150)
    print(f"📊 Summary charts saved to: {fig_path}")
    plt.show()


# ---------------------------------------------------------------------------
def main():
    debug_output_dir = "storage/sudoku/debug_solver"

    all_results = load_and_solve_all(DATASETS, debug_output_dir)

    if not all_results:
        print("No results produced — check the dataset paths in DATASETS.")
        return

    debug_print_boards(all_results, num_samples=5, seed=None)
    debug_show_confidence_heatmap(all_results, num_samples=4, seed=None)

    summary = generate_final_report(
        all_results,
        report_path=Path(debug_output_dir) / "final_report.txt",
    )
    debug_show_summary_charts(all_results, summary, output_dir=debug_output_dir)


if __name__ == "__main__":
    main()