from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def imread_color(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"Cannot write image: {path}")
    return str(path)


def resize_max(image: np.ndarray, max_side: int = 1400) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    scale = max_side / float(side)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
