from __future__ import annotations

from pathlib import Path
from typing import Any
import os

try:
    import yaml
except Exception:  # pragma: no cover - app can still start with defaults
    yaml = None  # type: ignore[assignment]


DEFAULT_CONFIG: dict[str, Any] = {
    "app": {"name": "Sudoku CV Windows API", "env": "development"},
    "server": {"host": "127.0.0.1", "port": 8000},
    "cors": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]},
    "paths": {
        "storage_dir": "storage",
        "uploads_dir": "storage/uploads",
        "runs_dir": "storage/runs",
        "models_dir": "models",
    },
    "preprocessing": {
        "max_side": 1400,
        "gaussian_blur_kernel": 7,
        "adaptive_threshold": {"block_size": 11, "c": 2},
    },
    "grid_detection": {
        "max_contours_to_check": 20,
        "min_contour_area": 10000,
        "approx_epsilon_ratio": 0.02,
        "board_size": 450,
    },
    "cell_extraction": {
        "margin_ratio": 0.14,
        "empty_pixel_ratio_threshold": 0.035,
        "digit_input_size": 28,
        "montage_cell_size": 40,
        "save_cells": True,
    },
    "digit_recognition": {
        "model_path": "models/digit_cnn.pth",
        "confidence_threshold": 0.75,
        "preprocessing": {"normalize_to_0_1": True, "invert_input": False},
        "model": {
            "name": "DigitCNN",
            "input_channels": 1,
            "num_classes": 10,
            "image_size": 28,
            "dropout": 0.25,
        },
        "training": {
            "seed": 42,
            "batch_size": 64,
            "epochs": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 6,
            "lr_scheduler_patience": 3,
            "lr_scheduler_factor": 0.5,
            "checkpoint_name": "digit_cnn.pth",
        },
        "data": {
            "validation_split": 0.1,
            "include_zero_digit": False,
            "num_workers": 0,
            "hoda": {
                "raw_dir": "phases/phase_04_digit_recognition/src/datasets/hoda",
                "train_cdb": "DigitDB/Train 60000.cdb",
                "test_cdb": "DigitDB/Test 20000.cdb",
            },
            "mnist": {
                "root_dir": "phases/phase_04_digit_recognition/src/datasets/mnist",
                "download": False,
            },
            "chars74k": {
                "root_dir": "phases/phase_04_digit_recognition/src/datasets/chars74k/EnglishFnt/Fnt",
                "test_split": 0.1,
            },
            "sudoku_cells": {
                "include_empty_cells": False,
                "validation_split": 0.1,
                "cache_dir": "storage/sudoku/cell_cache",
                "generated_roots": [
                    "phases/phase_04_digit_recognition/src/datasets/fromsoduko/sudoku_medium",
                ],
                "dat_roots": [
                    "storage/sudoku/raw/v2_train/v2_train",
                ],
            },
        },
        "sudoku_fine_tune": {
            "epochs": 5,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 3,
            "lr_scheduler_patience": 2,
            "lr_scheduler_factor": 0.5,
        },
        "augmentation": {
            "enabled": True,
            "pad_extra": 8,
            "affine_p": 0.35,
            "translate_percent": 0.03,
            "scale_min": 0.97,
            "scale_max": 1.03,
            "rotate_limit": 5,
            "shear_limit": 2,
            "brightness_limit": 0.08,
            "contrast_limit": 0.08,
            "brightness_contrast_p": 0.25,
            "blur_p": 0.04,
            "empty_threshold": 12,
        },
    },
    "solver": {"use_mrv": True, "empty_value": 0},
    "ui": {"show_cells_separately": True},
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


class Settings:
    """Project settings loaded from one YAML file: backend/config.yml."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / "config.yml"
        self.config = self._load_config()

        self.app_name = str(self.get("app.name", "Sudoku CV Windows API"))
        self.app_env = str(self.get("app.env", "development"))

        self.storage_dir = self.resolve_path(str(self.get("paths.storage_dir", "storage")))
        self.uploads_dir = self.resolve_path(str(self.get("paths.uploads_dir", "storage/uploads")))
        self.runs_dir = self.resolve_path(str(self.get("paths.runs_dir", "storage/runs")))
        self.models_dir = self.resolve_path(str(self.get("paths.models_dir", "models")))

        raw_origins = os.getenv("CORS_ORIGINS")
        if raw_origins:
            self._cors_origins = [item.strip() for item in raw_origins.split(",") if item.strip()]
        else:
            self._cors_origins = list(self.get("cors.origins", DEFAULT_CONFIG["cors"]["origins"]))

    def _load_config(self) -> dict[str, Any]:
        if yaml is None or not self.config_path.exists():
            return DEFAULT_CONFIG
        with self.config_path.open("r", encoding="utf-8") as file:
            user_config = yaml.safe_load(file) or {}
        if not isinstance(user_config, dict):
            raise ValueError("backend/config.yml must contain a YAML mapping at the top level.")
        return _deep_update(DEFAULT_CONFIG, user_config)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.config
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def resolve_path(self, relative_or_absolute: str) -> Path:
        path = Path(relative_or_absolute)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def cors_origins(self) -> list[str]:
        return self._cors_origins

    @property
    def digit_model_path(self) -> Path:
        raw = str(self.get("digit_recognition.model_path", "models/digit_cnn.pth"))
        return self.resolve_path(raw)


settings = Settings()
for directory in [settings.storage_dir, settings.uploads_dir, settings.runs_dir, settings.models_dir]:
    directory.mkdir(parents=True, exist_ok=True)
