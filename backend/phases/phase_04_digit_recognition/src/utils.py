from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import random
import numpy as np
from datetime import datetime

from app.config import settings


def phase4_root() -> Path:
    return Path(__file__).resolve().parents[1]


def phase4_data_dir() -> Path:
    return phase4_root() / "data"


def phase4_outputs_dir() -> Path:
    out = phase4_root() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def checkpoint_dir() -> Path:
    out = phase4_outputs_dir() / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    return out

def digit_runs_dir() -> Path:
    return settings.runs_dir / "digit_recognition"


def latest_run_dir(base_dir: Path | str | None = None) -> Path:

    base_dir = Path(base_dir) if base_dir is not None else digit_runs_dir()

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {base_dir}")

    run_dirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found inside: {base_dir}")

    return run_dirs[0]


def latest_checkpoint_path(
    checkpoint_filename: str | None = None,
    base_dir: Path | str | None = None,
) -> Path:

    if checkpoint_filename is None:
        checkpoint_filename = str(settings.get("digit_recognition.training.checkpoint_name", "digit_cnn.pth"))

    run_dir = latest_run_dir(base_dir)
    checkpoint_path = run_dir / checkpoint_filename

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found in latest run: {checkpoint_path}")

    return checkpoint_path


def reports_dir() -> Path:
    out = phase4_outputs_dir() / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_digit_config() -> dict[str, Any]:
    return settings.get("digit_recognition", {})


def set_seed(seed: int | None = None) -> int:
    if seed is None:
        seed = int(settings.get("digit_recognition.training.seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    return seed


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    

def save_training_plots(history: dict[str, Any], out_dir: Path | None = None) -> dict[str, str]:
    if out_dir is None:
        out_dir = reports_dir()

    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved_paths: dict[str, str] = {}

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss plot
    if history.get("train_loss") and history.get("val_loss"):
        plt.figure()
        plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
        plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)

        loss_path = out_dir / "loss_curve.png"
        plt.savefig(loss_path, dpi=200, bbox_inches="tight")
        plt.close()

        saved_paths["loss_curve"] = str(loss_path)

    # Accuracy plot
    if history.get("train_accuracy") and history.get("val_accuracy"):
        plt.figure()
        plt.plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
        plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)

        acc_path = out_dir / "accuracy_curve.png"
        plt.savefig(acc_path, dpi=200, bbox_inches="tight")
        plt.close()

        saved_paths["accuracy_curve"] = str(acc_path)

    return saved_paths



def new_run_dir(base_dir: Path | str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_dir = Path(base_dir) if base_dir is not None else digit_runs_dir()
    run_dir = root_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
