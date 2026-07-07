from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ClassificationMetrics:
    accuracy: float
    confusion_matrix: list[list[int]]


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / len(y_true)


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int = 10) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1
    return matrix.tolist()


def compute_metrics(y_true: list[int], y_pred: list[int], num_classes: int = 10) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred, num_classes=num_classes),
    )
