from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from app.config import settings
from common.images import save_image, resize_max


def preprocess_image(image_bgr: np.ndarray, output_dir: Path) -> dict[str, np.ndarray | str]:

    max_side = int(settings.get("preprocessing.max_side", 1400))
    blur_kernel = int(settings.get("preprocessing.gaussian_blur_kernel", 7))
    threshold_block = int(settings.get("preprocessing.adaptive_threshold.block_size", 11))
    threshold_c = int(settings.get("preprocessing.adaptive_threshold.c", 2))

    original = resize_max(image_bgr, max_side=max_side)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    threshold = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ## I think this is better/essential for open cv to find contours
        threshold_block,
        threshold_c,
    )

    save_image(output_dir / "01_original.png", original)
    save_image(output_dir / "02_grayscale.png", gray)
    save_image(output_dir / "03_blur.png", blur)
    save_image(output_dir / "04_threshold.png", threshold)

    return {
        "original": original,
        "gray": gray,
        "blur": blur,
        "threshold": threshold,
        "original_path": "01_original.png",
        "gray_path": "02_grayscale.png",
        "threshold_path": "04_threshold.png",
    }
