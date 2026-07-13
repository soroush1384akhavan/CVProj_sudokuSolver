from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def imread_color(path: str | Path) -> np.ndarray:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as image:
            orientation = image.getexif()

            image = image.convert("RGB")

            # Dataset-specific orientation fix
            # طبق تستی که انجام دادی:
            # orientation == 6  -> باید 90 درجه پادساعت‌گرد بچرخد
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(-90, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
            elif orientation == 5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                image = image.rotate(90, expand=True)
            elif orientation == 7:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                image = image.rotate(-90, expand=True)

            rgb = np.asarray(image)

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        return bgr.copy()

    except Exception as exc:
        raise ValueError(
            f"Could not read image: {path}"
        ) from exc


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
