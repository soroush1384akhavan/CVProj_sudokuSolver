# classifier.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings


@dataclass
class DigitPrediction:
    digit: int
    confidence: float
    used_model: bool


class PyTorchDigitClassifier:

    def __init__(self, model_path: Path | None = None, confidence_threshold: float | None = None, language: str = "en") -> None:
        self.language = language.strip().lower()

        if self.language not in ("en", "fa"):
            raise ValueError(f"Unsupported language: {self.language}. Choose 'en' or 'fa'.")

        self.model_path = model_path or self._resolve_model_path(self.language)
        print(f"[CLASSIFIER] language={self.language} | model_path={self.model_path}")
        self.confidence_threshold = float(
            confidence_threshold if confidence_threshold is not None else settings.get("digit_recognition.confidence_threshold", 0.75)
        )
        self.input_size = int(settings.get("digit_recognition.model.image_size", 28))
        self.invert_input = bool(settings.get("digit_recognition.preprocessing.invert_input", False))
        self.num_classes = int(settings.get("digit_recognition.model.num_classes", 10))
        self.dropout = float(settings.get("digit_recognition.model.dropout", 0.5))
        self._model = None
        self._torch = None
        self.status_message = "Model not loaded yet."

    @staticmethod
    def _resolve_model_path(language: str) -> Path:
        configured_path = settings.get(f"digit_recognition.model_paths.{language}", None)

        if configured_path:
            return settings.resolve_path(str(configured_path))

        try:
            from phases.phase_04_digit_recognition.src.utils import latest_checkpoint_path
            return latest_checkpoint_path()
        except Exception:
            return settings.digit_model_path

    def _load_model(self) -> bool:
        if self._model is not None:
            return True

        if not self.model_path.exists():
            self.status_message = f"No PyTorch model found at {self.model_path}. Returning empty board."
            return False

        try:
            # Import here so the code does not crash if no model is available.
            import torch
            from phases.phase_04_digit_recognition.src.model import build_model

            model = build_model(num_classes=self.num_classes, dropout=self.dropout)
            state = torch.load(self.model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()

            self._torch = torch
            self._model = model
            self.status_message = f"Loaded PyTorch model: {self.model_path.name}"
            return True
        except Exception as exc:
            self.status_message = f"Could not load PyTorch model: {exc}"
            return False

    def predict_cell(self, clean_binary: np.ndarray, empty_hint: bool) -> DigitPrediction:
        if empty_hint:
            return DigitPrediction(digit=0, confidence=1.0, used_model=False)

        if not self._load_model():
            return DigitPrediction(digit=0, confidence=0.0, used_model=False)

        torch = self._torch
        assert torch is not None and self._model is not None

        arr = clean_binary.astype("float32") / 255.0

        if self.invert_input:
            arr = 1.0 - arr

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, digit = torch.max(probs, dim=0)

        return DigitPrediction(digit=int(digit.item()), confidence=float(confidence.item()), used_model=True)

    def predict_board(
        self, clean_cells: list[np.ndarray], empty_flags: list[bool]
    )-> tuple[list[list[int]], list[list[float]], list[dict[str, int]]]:
        
        board = [[0 for _ in range(9)] for _ in range(9)]
        confidence = [[0.0 for _ in range(9)] for _ in range(9)]
        low_confidence_cells: list[dict[str, int]] = []

        for idx, cell in enumerate(clean_cells):
            r, c = divmod(idx, 9)
            pred = self.predict_cell(cell, empty_flags[idx])

            board[r][c] = pred.digit
            confidence[r][c] = round(pred.confidence, 4)

            if pred.digit != 0 and pred.confidence < self.confidence_threshold:
                low_confidence_cells.append({"row": r, "col": c})

            if not pred.used_model and not empty_flags[idx]:
                low_confidence_cells.append({"row": r, "col": c})

        return board, confidence, low_confidence_cells
