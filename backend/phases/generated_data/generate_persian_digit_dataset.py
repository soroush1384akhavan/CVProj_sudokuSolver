from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PERSIAN_DIGITS = {
    0: "۰",
    1: "۱",
    2: "۲",
    3: "۳",
    4: "۴",
    5: "۵",
    6: "۶",
    7: "۷",
    8: "۸",
    9: "۹",
}

FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}

BAD_FONT_KEYWORDS = [
    "wingdings",
    "webdings",
    "symbol",
    "emoji",
    "icon",
    "icons",
    "fontawesome",
    "materialicons",
    "segoe mdl2 assets",
    "segoe fluent icons",
]

PREFERRED_FA_KEYWORDS = [
    "vazir",
    "vazirmatn",
    "iransans",
    "nazanin",
    "shabnam",
    "sahel",
    "yekan",
    "tahoma",
    "arial",
]

SYSTEM_FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local/share/fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
]


def glyph_signature(font: ImageFont.FreeTypeFont, text: str) -> bytes | None:
    canvas = Image.new("L", (96, 96), 0)
    draw = ImageDraw.Draw(canvas)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except Exception:
        return None

    if bbox is None:
        return None

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    if width <= 0 or height <= 0:
        return None

    x = (96 - width) // 2 - bbox[0]
    y = (96 - height) // 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)

    content_bbox = canvas.getbbox()
    if content_bbox is None:
        return None

    return canvas.crop(content_bbox).resize((32, 32)).tobytes()


def supports_persian_digits(font_path: Path, size: int = 52) -> bool:
    try:
        font = ImageFont.truetype(str(font_path), size=size)
    except Exception:
        return False

    signatures = []

    for digit in PERSIAN_DIGITS.values():
        signature = glyph_signature(font, digit)
        if signature is None:
            return False
        signatures.append(signature)

    return len(set(signatures)) >= 8


def font_score(font_path: Path) -> int:
    name = font_path.stem.lower()
    return sum(keyword in name for keyword in PREFERRED_FA_KEYWORDS)


def collect_fonts(font_dir: str, include_system_fonts: bool = True) -> list[Path]:
    raw_dir = Path(font_dir)

    if raw_dir.is_absolute():
        candidates = [raw_dir]
    else:
        candidates = [
            Path.cwd() / raw_dir,
            Path(__file__).resolve().parent / raw_dir,
        ]

    if include_system_fonts:
        candidates.extend(SYSTEM_FONT_DIRS)

    fonts: list[Path] = []
    seen: set[str] = set()

    for directory in candidates:
        directory = directory.resolve()

        if not directory.is_dir():
            continue

        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in FONT_EXTENSIONS:
                continue

            name = path.stem.lower()

            if any(keyword in name for keyword in BAD_FONT_KEYWORDS):
                continue

            key = str(path.resolve())

            if key not in seen:
                seen.add(key)
                fonts.append(path)

    return fonts


def select_persian_fonts(font_dir: str, font_count: int = 20) -> list[Path]:
    candidates = collect_fonts(font_dir)
    candidates.sort(key=lambda path: (-font_score(path), path.name.lower()))

    selected: list[Path] = []

    for path in candidates:
        if supports_persian_digits(path):
            selected.append(path)
            print(f"[FONT] {len(selected):02d}/{font_count}: {path.name}")

        if len(selected) >= font_count:
            break

    if len(selected) < font_count:
        raise RuntimeError(
            f"Only {len(selected)} valid Persian fonts were found. "
            f"Put at least {font_count} Persian fonts in: {font_dir}"
        )

    return selected


def render_digit(
    digit: int,
    font_path: Path,
    image_size: int,
    font_scale: float,
) -> Image.Image:
    image = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(image)

    font_size = max(8, int(image_size * font_scale))
    font = ImageFont.truetype(str(font_path), size=font_size)
    text = PERSIAN_DIGITS[digit]

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image_size - text_width) / 2 - bbox[0]
    y = (image_size - text_height) / 2 - bbox[1]

    draw.text((x, y), text, fill=255, font=font)
    return image


def generate_dataset(
    out_root: str = "generated_digits",
    samples_per_digit: int = 1000,
    image_size: int = 64,
    font_dir: str = "Fonts",
    font_count: int = 20,
    min_font_scale: float = 0.55,
    max_font_scale: float = 0.78,
    seed: int | None = None,
) -> Path:
    if samples_per_digit < 1:
        raise ValueError("samples_per_digit must be at least 1.")

    if image_size < 16:
        raise ValueError("image_size must be at least 16.")

    if font_count != 20:
        raise ValueError("This generator is configured to use exactly 20 fonts.")

    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)

    random.seed(seed)

    fonts = select_persian_fonts(font_dir=font_dir, font_count=font_count)

    run_name = datetime.now().strftime(f"persian_digits_%Y%m%d_%H%M%S_seed_{seed}")
    out_dir = Path(out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    labels_path = out_dir / "labels.csv"

    with labels_path.open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(
            labels_file,
            fieldnames=["filename", "label", "persian_digit", "font"],
        )
        writer.writeheader()

        for digit in range(10):
            class_dir = out_dir / str(digit)
            class_dir.mkdir(parents=True, exist_ok=True)

            for index in range(samples_per_digit):
                font_path = fonts[index % len(fonts)]
                font_scale = random.uniform(min_font_scale, max_font_scale)

                image = render_digit(
                    digit=digit,
                    font_path=font_path,
                    image_size=image_size,
                    font_scale=font_scale,
                )

                filename = f"{digit}_{index:06d}.png"
                output_path = class_dir / filename
                image.save(output_path)

                writer.writerow({
                    "filename": str(output_path.relative_to(out_dir)),
                    "label": digit,
                    "persian_digit": PERSIAN_DIGITS[digit],
                    "font": font_path.name,
                })

            print(f"[DONE] digit={digit} | samples={samples_per_digit}")

    metadata = {
        "language": "fa",
        "background": "black",
        "foreground": "white",
        "digits": PERSIAN_DIGITS,
        "samples_per_digit": samples_per_digit,
        "total_samples": samples_per_digit * 10,
        "image_size": image_size,
        "font_count": len(fonts),
        "fonts": [font.name for font in fonts],
        "seed": seed,
        "output_directory": str(out_dir),
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    print()
    print(f"Output directory: {out_dir}")
    print(f"Total samples: {samples_per_digit * 10}")
    print(f"Fonts used: {len(fonts)}")

    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate white Persian digits centered on black square images."
    )
    parser.add_argument("--out-root", type=str, default="generated_digits")
    parser.add_argument("--samples-per-digit", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--font-dir", type=str, default="Fonts")
    parser.add_argument("--font-count", type=int, default=20)
    parser.add_argument("--min-font-scale", type=float, default=0.55)
    parser.add_argument("--max-font-scale", type=float, default=0.78)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_dataset(
        out_root=args.out_root,
        samples_per_digit=args.samples_per_digit,
        image_size=args.image_size,
        font_dir=args.font_dir,
        font_count=args.font_count,
        min_font_scale=args.min_font_scale,
        max_font_scale=args.max_font_scale,
        seed=args.seed,
    )