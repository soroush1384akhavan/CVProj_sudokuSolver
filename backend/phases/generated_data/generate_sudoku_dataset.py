from __future__ import annotations

import argparse
import csv
import io
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


BASE = 3
SIDE = 9

SCRIPT_DIR = Path(__file__).resolve().parent


def safe_parent(path: Path, levels: int) -> Path:
    try:
        return path.parents[levels - 1]
    except IndexError:
        return path.parent



DIFFICULTY_LEVELS = {
    "easy": 35,
    "medium": 45,
    "hard": 55,
    "expert": 60,
}

PERSIAN_DIGITS = {
    "0": "۰",
    "1": "۱",
    "2": "۲",
    "3": "۳",
    "4": "۴",
    "5": "۵",
    "6": "۶",
    "7": "۷",
    "8": "۸",
    "9": "۹",
}

LABEL_FIELDS = [
    "filename",
    "style",
    "kind",
    "puzzle",
    "solution",
    "difficulty",
    "requested_blanks",
    "actual_blanks",
    "seed",
]

FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}

BAD_FONT_KEYWORDS = [
    "wingdings",
    "webdings",
    "symbol",
    "seguisym",
    "segoe fluent icons",
    "segoe mdl2 assets",
    "materialicons",
    "fontawesome",
    "emoji",
    "icons",
    "icon",
    "dingbats",
]

PREFERRED_FA_KEYWORDS = [
    "vazir",
    "vazirmatn",
    "iransans",
    "nazanin",
    "tahoma",
    "arial",
    "shabnam",
    "sahel",
    "yekan",
]

PREFERRED_EN_KEYWORDS = [
    "arial",
    "tahoma",
    "dejavu",
    "times",
    "calibri",
    "verdana",
    "segoe",
    "roboto",
    "open sans",
    "opensans",
    "lato",
    "montserrat",
    "ubuntu",
    "courier",
    "georgia",
    "cambria",
    "candara",
    "consola",
    "liberation",
    "source sans",
    "sourcesans",
    "inter",
    "helvetica",
    "myriad",
    "garamond",
]

ENGLISH_EXCLUDED_FONT_KEYWORDS = [
    "azarmehr",
    "vazir",
    "vazirmatn",
    "iransans",
    "nazanin",
    "shabnam",
    "sahel",
    "yekan",
    "persian",
    "farsi",
    "arabic",
    "urdu",
]

ENGLISH_CURATED_FONT_KEYWORDS = [
    "arial",
    "calibri",
    "verdana",
    "tahoma",
    "timesnewroman",
    "times new roman",
    "times",
    "georgia",
    "cambria",
    "candara",
    "couriernew",
    "courier new",
    "consola",
    "consolas",
    "segoeui",
    "segoe ui",
    "dejavusans",
    "dejavu sans",
    "dejavuserif",
    "dejavu serif",
    "liberationsans",
    "liberation sans",
    "liberationserif",
    "liberation serif",
    "roboto",
    "opensans",
    "open sans",
    "lato",
    "ubuntu",
    "inter",
]

SYSTEM_FONT_DIR_CANDIDATES = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local/share/fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
]


# ---------------------------
# Sudoku generation
# ---------------------------

def pattern(r: int, c: int) -> int:
    return (BASE * (r % BASE) + r // BASE + c) % SIDE


def shuffled(seq):
    seq = list(seq)
    random.shuffle(seq)
    return seq


def generate_solution():
    rows = [g * BASE + r for g in shuffled(range(BASE)) for r in shuffled(range(BASE))]
    cols = [g * BASE + c for g in shuffled(range(BASE)) for c in shuffled(range(BASE))]
    nums = shuffled(range(1, 10))
    return [[nums[pattern(r, c)] for c in cols] for r in rows]


def count_solutions(board, limit=2):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empties = []

    for r in range(9):
        for c in range(9):
            n = board[r][c]
            if n == 0:
                empties.append((r, c))
            else:
                b = (r // 3) * 3 + c // 3
                if n in rows[r] or n in cols[c] or n in boxes[b]:
                    return 0
                rows[r].add(n)
                cols[c].add(n)
                boxes[b].add(n)

    nums = set(range(1, 10))
    count = 0

    def backtrack():
        nonlocal count

        if count >= limit:
            return

        best_pos = None
        best_cands = None

        for r, c in empties:
            if board[r][c] != 0:
                continue

            b = (r // 3) * 3 + c // 3
            cands = list(nums - rows[r] - cols[c] - boxes[b])

            if best_cands is None or len(cands) < len(best_cands):
                best_pos = (r, c)
                best_cands = cands

                if len(cands) == 0:
                    return

        if best_pos is None:
            count += 1
            return

        r, c = best_pos
        b = (r // 3) * 3 + c // 3
        random.shuffle(best_cands)

        for n in best_cands:
            board[r][c] = n
            rows[r].add(n)
            cols[c].add(n)
            boxes[b].add(n)

            backtrack()

            rows[r].remove(n)
            cols[c].remove(n)
            boxes[b].remove(n)
            board[r][c] = 0

            if count >= limit:
                return

    backtrack()
    return count


def make_puzzle(solution, blanks=45):
    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)

    removed = 0

    for r, c in cells:
        if removed >= blanks:
            break

        old = puzzle[r][c]
        puzzle[r][c] = 0

        test_board = [row[:] for row in puzzle]

        if count_solutions(test_board, limit=2) == 1:
            removed += 1
        else:
            puzzle[r][c] = old

    return puzzle


def flatten_board(board):
    return "".join(str(board[r][c]) for r in range(9) for c in range(9))


def to_digit(n: int, style: str):
    text = str(n)
    if style == "fa":
        return "".join(PERSIAN_DIGITS[ch] for ch in text)
    return text


# ---------------------------
# Font utilities
# ---------------------------

def is_bad_font_file(font_path):
    name = Path(font_path).stem.lower()
    return any(k in name for k in BAD_FONT_KEYWORDS)


def collect_font_paths(font_dir="Fonts", include_system_fonts=True):
    """
    مسیر فونت‌ها را مقاوم‌تر پیدا می‌کند.
    علاوه بر --font-dir، در صورت نیاز فونت‌های سیستمی رایج را هم بررسی می‌کند
    تا برای انگلیسی بتوان از فونت‌های معروف و استاندارد استفاده کرد.
    """
    font_paths = []
    seen = set()

    raw_font_dir = Path(font_dir)

    if raw_font_dir.is_absolute():
        candidates = [raw_font_dir]
    else:
        candidates = [
            Path.cwd() / raw_font_dir,
            SCRIPT_DIR / raw_font_dir,
            safe_parent(SCRIPT_DIR, 1) / raw_font_dir,
            safe_parent(SCRIPT_DIR, 2) / raw_font_dir,
            safe_parent(SCRIPT_DIR, 3) / raw_font_dir,
        ]

    if include_system_fonts:
        candidates.extend(SYSTEM_FONT_DIR_CANDIDATES)

    unique_candidates = []
    seen_candidates = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        key = str(candidate)
        if key not in seen_candidates:
            seen_candidates.add(key)
            unique_candidates.append(candidate)

    print("[FONT] cwd:", Path.cwd())
    print("[FONT] candidates:")
    for candidate in unique_candidates:
        print("   ", candidate, "| exists:", candidate.exists())

    for font_dir_path in unique_candidates:
        if not font_dir_path.exists() or not font_dir_path.is_dir():
            continue

        for path in font_dir_path.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in FONT_EXTENSIONS:
                continue

            if is_bad_font_file(path):
                print("[FONT] skipped bad/icon font:", path.name)
                continue

            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                font_paths.append(path)

    print("[FONT] total candidate fonts:", len(font_paths))
    return font_paths


def glyph_signature(font, text):
    canvas = Image.new("L", (80, 80), 0)
    draw = ImageDraw.Draw(canvas)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except Exception:
        return None

    if bbox is None:
        return None

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    if w <= 0 or h <= 0:
        return None

    x = (80 - w) // 2 - bbox[0]
    y = (80 - h) // 2 - bbox[1]

    draw.text((x, y), text, fill=255, font=font)

    cropped_bbox = canvas.getbbox()
    if cropped_bbox is None:
        return None

    cropped = canvas.crop(cropped_bbox).resize((32, 32))
    return cropped.tobytes()


def font_supports_digits_safely(font_path, style, size=42):
    try:
        font = ImageFont.truetype(str(font_path), size=size)
    except Exception:
        return False

    digits = "۱۲۳۴۵۶۷۸۹" if style == "fa" else "123456789"
    signatures = []

    for ch in digits:
        sig = glyph_signature(font, ch)
        if sig is None:
            return False
        signatures.append(sig)

    unique_count = len(set(signatures))
    if unique_count < 5:
        return False

    return True


def count_different_digit_pairs(font, digits_a, digits_b):
    different_pairs = 0
    comparable_pairs = 0

    for ch_a, ch_b in zip(digits_a, digits_b):
        sig_a = glyph_signature(font, ch_a)
        sig_b = glyph_signature(font, ch_b)

        if sig_a is None or sig_b is None:
            continue

        comparable_pairs += 1
        if sig_a != sig_b:
            different_pairs += 1

    return different_pairs, comparable_pairs


def is_farsi_digit_font_name(font_path):
    name = Path(font_path).stem.lower()
    normalized = name.replace("_", "-").replace(" ", "-")
    tokens = [token for token in normalized.split("-") if token]

    return (
        "fd" in tokens
        or "farsidigit" in normalized
        or "farsi-digit" in normalized
        or "persiandigit" in normalized
        or "persian-digit" in normalized
    )


def is_english_font_candidate(font_path, size=42):
    name = Path(font_path).stem.lower()

    if any(keyword in name for keyword in ENGLISH_EXCLUDED_FONT_KEYWORDS):
        return False

    if is_farsi_digit_font_name(font_path):
        return False

    # فقط فونت‌های انگلیسی معروف/استاندارد پذیرفته شوند.
    if not any(keyword in name for keyword in ENGLISH_CURATED_FONT_KEYWORDS):
        return False

    if not font_supports_digits_safely(font_path, style="en", size=size):
        return False

    try:
        font = ImageFont.truetype(str(font_path), size=size)
    except Exception:
        return False

    different_pairs, comparable_pairs = count_different_digit_pairs(
        font,
        "123456789",
        "۱۲۳۴۵۶۷۸۹",
    )

    # اگر فونت برای ارقام انگلیسی و فارسی تقریباً یک شکل بدهد،
    # احتمالاً در خروجی انگلیسی ظاهر فارسی/ترکیبی ایجاد می‌کند.
    if comparable_pairs >= 5 and different_pairs < 5:
        return False

    return True


def is_persian_font_candidate(font_path, size=42):
    return font_supports_digits_safely(font_path, style="fa", size=size)


def score_font_for_style(font_path, style):
    name = Path(font_path).stem.lower()
    keywords = PREFERRED_FA_KEYWORDS if style == "fa" else PREFERRED_EN_KEYWORDS

    score = 0
    for kw in keywords:
        if kw in name:
            score += 1
    return score


def filter_fonts_for_style(
    font_paths,
    style,
    size=42,
    max_fonts=10,
):
    """
    فونت‌ها را بر اساس اولویت زبان مرتب می‌کند و فقط تا سقف max_fonts
    فونت معتبر برای همان زبان نگه می‌دارد.
    """
    if max_fonts < 1:
        raise ValueError("max_fonts must be at least 1.")

    valid_fonts = []

    print(f"[FONT] filtering style={style} | limit={max_fonts}")

    ordered_paths = sorted(
        font_paths,
        key=lambda p: (
            -score_font_for_style(p, style),
            Path(p).name.lower(),
        ),
    )

    for path in ordered_paths:
        if len(valid_fonts) >= max_fonts:
            break

        if is_bad_font_file(path):
            print(f"[FONT] rejected bad/icon font for {style}: {Path(path).name}")
            continue

        if style == "en":
            accepted = is_english_font_candidate(path, size=size)
        else:
            accepted = is_persian_font_candidate(path, size=size)

        if accepted:
            valid_fonts.append(str(path))
            print(
                f"[FONT] accepted for {style} "
                f"({len(valid_fonts)}/{max_fonts}): {Path(path).name}"
            )
        else:
            print(f"[FONT] rejected for {style}: {Path(path).name}")

    print(f"[FONT] selected for {style}: {len(valid_fonts)}")
    return valid_fonts


def prepare_font_pools(font_dir="Fonts", max_fonts_per_language=10):
    all_font_paths = collect_font_paths(font_dir=font_dir)

    en_fonts = filter_fonts_for_style(
        all_font_paths,
        style="en",
        size=42,
        max_fonts=max_fonts_per_language,
    )
    fa_fonts = filter_fonts_for_style(
        all_font_paths,
        style="fa",
        size=42,
        max_fonts=max_fonts_per_language,
    )

    return {
        "en": en_fonts,
        "fa": fa_fonts,
    }


def choose_random_font(style="en", font_pools=None, size=42):
    if font_pools is None:
        font_pools = {"en": [], "fa": []}

    pool = font_pools.get(style, [])

    # فارسی: از بین فونت‌های accepted به‌صورت رندوم انتخاب می‌شود.
    # انگلیسی: از بین فونت‌های معروف/استانداردی که در pool جمع‌آوری شده‌اند.
    if pool:
        chosen_path = random.choice(pool)
        try:
            return ImageFont.truetype(chosen_path, size=size)
        except Exception:
            pass

    fallback_candidates = []

    if style == "fa":
        fallback_candidates = [
            "Fonts/Vazirmatn-Regular.ttf",
            "Fonts/Vazirmatn-Bold.ttf",
            "Fonts/IRANSans.ttf",
            "Fonts/Tahoma.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    else:
        fallback_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/times.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "Fonts/Arial.ttf",
            "Fonts/Calibri.ttf",
            "Fonts/Verdana.ttf",
            "Fonts/Tahoma.ttf",
        ]

    for path in fallback_candidates:
        p = Path(path)
        if not p.exists():
            continue
        try:
            return ImageFont.truetype(str(p), size=size)
        except Exception:
            pass

    return ImageFont.load_default()


# ---------------------------
# Rendering
# ---------------------------

def render_sudoku(
    board,
    out_path,
    style="en",
    cell=64,
    margin=32,
    font_pools=None,
):
    size = cell * 9 + margin * 2

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    font_size = int(cell * 0.58)
    font = choose_random_font(style=style, font_pools=font_pools, size=font_size)

    for i in range(10):
        width = 5 if i % 3 == 0 else 2

        x = margin + i * cell
        y = margin + i * cell

        draw.line((x, margin, x, margin + 9 * cell), fill="black", width=width)
        draw.line((margin, y, margin + 9 * cell, y), fill="black", width=width)

    for r in range(9):
        for c in range(9):
            n = board[r][c]
            if n == 0:
                continue

            text = to_digit(n, style)
            bbox = draw.textbbox((0, 0), text, font=font)

            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            x = margin + c * cell + (cell - text_w) / 2 - bbox[0]
            y = margin + r * cell + (cell - text_h) / 2 - bbox[1]

            draw.text((x, y), text, fill="black", font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, optimize=True)

    return img




# ---------------------------
# Strong image variations
# ---------------------------

def find_perspective_coeffs(dst_pts, src_pts):
    matrix = []
    vector = []

    for (x, y), (u, v) in zip(dst_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])

    A = np.array(matrix, dtype=np.float64)
    B = np.array(vector, dtype=np.float64)

    coeffs, *_ = np.linalg.lstsq(A, B, rcond=None)
    return coeffs.tolist()


def add_padding(img, pad_ratio=0.12):
    img = img.convert("RGB")
    w, h = img.size
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)

    canvas = Image.new("RGB", (w + 2 * pad_x, h + 2 * pad_y), "white")
    canvas.paste(img, (pad_x, pad_y))
    return canvas


def apply_strong_perspective(img, distortion=0.10):
    """
    Perspective امن: گوشه‌های مقصد داخل canvas می‌مانند.
    نسخه قبلی اجازه می‌داد بعضی گوشه‌ها با مختصات منفی یا بزرگ‌تر از عرض/ارتفاع بروند
    و نتیجه‌اش خروج جدول از صفحه بود.
    """
    img = img.convert("RGB")
    w, h = img.size

    max_dx = w * distortion
    max_dy = h * distortion

    src_pts = [(0, 0), (w, 0), (w, h), (0, h)]

    dst_pts = [
        (random.uniform(0, max_dx), random.uniform(0, max_dy)),
        (w - random.uniform(0, max_dx), random.uniform(0, max_dy)),
        (w - random.uniform(0, max_dx), h - random.uniform(0, max_dy)),
        (random.uniform(0, max_dx), h - random.uniform(0, max_dy)),
    ]

    coeffs = find_perspective_coeffs(dst_pts, src_pts)

    return img.transform(
        img.size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )


def apply_affine_shear(img, strength=0.08):
    img = img.convert("RGB")
    w, h = img.size

    shear_x = random.uniform(-strength, strength)
    shear_y = random.uniform(-strength, strength)

    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, shear_x, 0, shear_y, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )


def add_gaussian_noise(img, sigma=14.0):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.int16)

    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def add_salt_pepper_noise(img, amount=0.012):
    img = img.convert("RGB")
    arr = np.array(img)

    h, w, c = arr.shape
    num_pixels = int(h * w * amount)

    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)

    values = np.random.choice([0, 255], size=(num_pixels, 1))
    arr[ys, xs] = values

    return Image.fromarray(arr)


def add_paper_texture(img, strength=18):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.int16)

    h, w, _ = arr.shape

    texture = np.random.normal(0, strength, (h, w, 1))
    texture = np.repeat(texture, 3, axis=2)

    arr = np.clip(arr + texture, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def add_uneven_lighting(img, strength=55):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)

    h, w, _ = arr.shape

    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)

    cx = random.uniform(-0.7, 0.7)
    cy = random.uniform(-0.7, 0.7)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = 1.0 - np.clip(dist, 0, 1)

    if random.random() < 0.5:
        mask = -mask

    lighting = mask[..., None] * strength

    arr = np.clip(arr + lighting, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def add_soft_shadow(img, alpha_max=55):
    base = img.convert("RGBA")
    w, h = base.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    shape_type = random.choice(["ellipse", "rectangle"])

    x0 = random.randint(-w // 3, w // 2)
    y0 = random.randint(-h // 3, h // 2)
    x1 = x0 + random.randint(w // 2, int(w * 1.4))
    y1 = y0 + random.randint(h // 2, int(h * 1.4))

    alpha = random.randint(18, alpha_max)

    if shape_type == "ellipse":
        draw.ellipse((x0, y0, x1, y1), fill=(0, 0, 0, alpha))
    else:
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, alpha))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.uniform(35, 90)))

    return Image.alpha_composite(base, overlay).convert("RGB")


def add_jpeg_artifact(img, quality_range=(25, 65)):
    img = img.convert("RGB")

    buffer = io.BytesIO()
    quality = random.randint(*quality_range)

    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


def add_motion_blur(img):
    img = img.convert("RGB")

    kernel_size = random.choice([3, 5])

    if random.random() < 0.5:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
    else:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[:, kernel_size // 2] = 1.0 / kernel_size

    kernel_list = kernel.flatten().tolist()

    try:
        return img.filter(
            ImageFilter.Kernel(
                size=(kernel_size, kernel_size),
                kernel=kernel_list,
                scale=sum(kernel_list),
                offset=0,
            )
        )
    except Exception:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.4)))


def degrade_resolution(img, scale_min=0.45, scale_max=0.85):
    img = img.convert("RGB")
    w, h = img.size

    scale = random.uniform(scale_min, scale_max)

    small_w = max(32, int(w * scale))
    small_h = max(32, int(h * scale))

    img = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
    img = img.resize((w, h), Image.Resampling.BILINEAR)

    return img


def random_crop_and_resize(img, crop_ratio=0.04):
    img = img.convert("RGB")
    w, h = img.size

    max_dx = int(w * crop_ratio)
    max_dy = int(h * crop_ratio)

    left = random.randint(0, max_dx)
    top = random.randint(0, max_dy)
    right = w - random.randint(0, max_dx)
    bottom = h - random.randint(0, max_dy)

    img = img.crop((left, top, right, bottom))
    img = img.resize((w, h), Image.Resampling.BICUBIC)

    return img


def make_image_variant(img, severity=1.6, always_add_noise=False):
    """
    Augmentation قوی‌تر برای ساخت دیتاست متنوع‌تر.
    severity حدود پیشنهادی:
    1.0 = ملایم
    1.5 = متوسط رو به قوی
    2.0 = قوی
    2.5 = خیلی قوی و ممکن است بعضی تصاویر سخت‌خوان شوند
    """

    original_size = img.size

    img = img.convert("RGB")

    # برای اینکه rotation/perspective گوشه‌ها را زیاد نبرد، اول حاشیه می‌دهیم
    img = add_padding(img, pad_ratio=random.uniform(0.08, 0.16))

    # نور و کنتراست قوی‌تر
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.72, 1.28))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.65, 1.45))

    # رنگ سفید کاغذ را کمی طبیعی‌تر می‌کند
    if random.random() < 0.70:
        img = add_paper_texture(img, strength=random.uniform(6, 18) * severity)

    # نور ناهموار
    if random.random() < 0.75:
        img = add_uneven_lighting(img, strength=random.uniform(25, 65) * severity)

    # projection / perspective قوی‌تر
    if random.random() < 0.95:
        img = apply_strong_perspective(
            img,
            distortion=random.uniform(0.055, 0.12) * severity
        )

    # shear / affine
    if random.random() < 0.75:
        img = apply_affine_shear(
            img,
            strength=random.uniform(0.025, 0.08) * severity
        )

    # rotation قوی‌تر
    angle = random.uniform(-8.0, 8.0) * severity
    img = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )

    # crop خیلی محدود؛ crop زیاد باعث می‌شود جدول از کادر بزند بیرون
    if random.random() < 0.12:
        img = random_crop_and_resize(
            img,
            crop_ratio=random.uniform(0.004, 0.012)
        )

    # shadow واضح‌تر
    if random.random() < 0.75:
        img = add_soft_shadow(img, alpha_max=int(35 + 18 * severity))

    # blur قوی‌تر: تمرکز روی تاری واقعی به‌جای نویز
    if random.random() < 0.85:
        img = img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.8, 2.4) * severity)
        )

    # motion blur بیشتر برای نمونه‌های تارتر
    if random.random() < 0.55:
        img = add_motion_blur(img)
        if random.random() < 0.40:
            img = add_motion_blur(img)

    # کاهش کیفیت رزولوشن برای ایجاد blur طبیعی‌تر
    if random.random() < 0.85:
        img = degrade_resolution(
            img,
            scale_min=max(0.20, 0.55 - 0.15 * severity),
            scale_max=0.82,
        )

    # نویز دیگر اجباری نیست؛ فقط گاهی و خیلی ملایم
    if always_add_noise:
        img = add_gaussian_noise(img, sigma=random.uniform(4.0, 8.0) * severity)
    else:
        if random.random() < 0.15:
            img = add_gaussian_noise(img, sigma=random.uniform(3.0, 6.0) * severity)

        # salt pepper noise بسیار کم، فقط برای تنوع محدود
        if random.random() < 0.05:
            img = add_salt_pepper_noise(
                img,
                amount=random.uniform(0.0005, 0.0015)
            )
    # jpeg artifacts
    if random.random() < 0.65:
        img = add_jpeg_artifact(
            img,
            quality_range=(25, 70)
        )

    # خروجی را دوباره به سایز اصلی برمی‌گردانیم
    img = img.resize(original_size, Image.Resampling.BICUBIC)

    return img


def content_touches_border(img, threshold=140, border=18):
    """
    اگر محتوای تیره‌ی تصویر خیلی به لبه نزدیک باشد، احتمالاً جدول از کادر خارج شده
    یا بعداً برای cell extraction دردسر درست می‌کند.
    """
    gray = img.convert("L")
    arr = np.array(gray)

    mask = arr < threshold

    if mask.sum() < 50:
        return True

    ys, xs = np.where(mask)

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())

    w, h = img.size

    return (
        min_x < border or
        min_y < border or
        max_x > w - border or
        max_y > h - border
    )


def make_safe_image_variant(
    original_img,
    severity=1.2,
    always_add_noise=False,
    max_attempts=12,
    border=18,
    threshold=140,
):
    """
    چند بار variant می‌سازد و فقط وقتی ذخیره می‌کند که جدول به لبه‌ها نچسبیده باشد.
    اگر همه تلاش‌ها بد شد، یک variant خیلی ملایم می‌سازد تا تصویر از کادر خارج نشود.
    """
    for attempt in range(max_attempts):
        candidate = make_image_variant(
            original_img,
            severity=severity,
            always_add_noise=always_add_noise,
        )

        if not content_touches_border(candidate, threshold=threshold, border=border):
            return candidate

    # fallback بسیار امن
    candidate = make_image_variant(
        original_img,
        severity=0.75,
        always_add_noise=always_add_noise,
    )

    if not content_touches_border(candidate, threshold=threshold, border=border):
        return candidate

    # آخرین راه: خود تصویر اصلی + نویز خیلی ملایم، چون خود جدول اصلی حاشیه دارد
    return add_gaussian_noise(original_img, sigma=6.0)

# ---------------------------
# CSV helpers
# ---------------------------

def init_labels_csv(labels_path: Path):
    f = open(labels_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
    writer.writeheader()
    f.flush()
    return f, writer


def append_label(writer, file_obj, row):
    writer.writerow(row)
    file_obj.flush()


# ---------------------------
# Dataset generation
# ---------------------------

def generate_dataset(
    out_root="sudoku_runs",
    count=50,
    difficulty="medium",
    blanks=None,
    variants_per_image=3,
    font_dir="Fonts",
    max_fonts_per_language=10,
    seed=None,
    variant_severity=1.2,
    max_variant_attempts=12,
    border_check=18,
):
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(
            f"Invalid difficulty: {difficulty}. Choose from: {list(DIFFICULTY_LEVELS.keys())}"
        )

    if blanks is None:
        blanks = DIFFICULTY_LEVELS[difficulty]

    if blanks < 0 or blanks > 80:
        raise ValueError("blanks must be between 0 and 80.")

    if max_fonts_per_language < 1:
        raise ValueError("max_fonts_per_language must be at least 1.")

    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)

    print("Preparing font pools...")
    font_pools = prepare_font_pools(
        font_dir=font_dir,
        max_fonts_per_language=max_fonts_per_language,
    )

    print(f"English fonts selected: {len(font_pools['en'])}")
    print(f"Persian fonts selected: {len(font_pools['fa'])}")

    if len(font_pools["fa"]) == 0:
        print("WARNING: No safe Persian font found in Fonts folder.")

    if len(font_pools["en"]) == 0:
        print("WARNING: No safe English font found in Fonts folder.")

    run_name = datetime.now().strftime(
        f"sudoku_{difficulty}_%Y%m%d_%H%M%S_%f_seed_{seed}"
    )

    out_dir = Path(out_root) / run_name
    metadata_path = out_dir / "metadata.json"

    out_dir.mkdir(parents=True, exist_ok=False)

    styles = ("en", "fa")
    style_outputs = {}
    label_resources = {}
    total_written_by_style = {style: 0 for style in styles}

    for style in styles:
        style_dir = out_dir / style
        images_original_dir = style_dir / "images_original"
        images_variant_dir = style_dir / "images_variant"
        labels_path = style_dir / "labels.csv"

        images_original_dir.mkdir(parents=True, exist_ok=True)
        images_variant_dir.mkdir(parents=True, exist_ok=True)

        labels_file, writer = init_labels_csv(labels_path)
        label_resources[style] = (labels_file, writer)

        style_outputs[style] = {
            "directory": style_dir,
            "images_original": images_original_dir,
            "images_variant": images_variant_dir,
            "labels": labels_path,
        }

    try:
        for idx in range(count):
            solution = generate_solution()
            puzzle = make_puzzle(solution, blanks=blanks)

            actual_blanks = sum(
                1 for r in range(9) for c in range(9) if puzzle[r][c] == 0
            )

            puzzle_flat = flatten_board(puzzle)
            solution_flat = flatten_board(solution)

            for style in styles:
                style_info = style_outputs[style]
                labels_file, writer = label_resources[style]
                base_name = f"sudoku_{idx:05d}_{style}"

                original_path = style_info["images_original"] / f"{base_name}.png"
                original_img = render_sudoku(
                    puzzle,
                    original_path,
                    style=style,
                    font_pools=font_pools,
                )

                row = {
                    "filename": str(original_path.relative_to(style_info["directory"])),
                    "style": style,
                    "kind": "original",
                    "puzzle": puzzle_flat,
                    "solution": solution_flat,
                    "difficulty": difficulty,
                    "requested_blanks": blanks,
                    "actual_blanks": actual_blanks,
                    "seed": seed,
                }
                append_label(writer, labels_file, row)
                total_written_by_style[style] += 1

                for var_idx in range(variants_per_image):
                    variant_img = make_safe_image_variant(
                        original_img,
                        severity=variant_severity,
                        always_add_noise=False,
                        max_attempts=max_variant_attempts,
                        border=border_check,
                        threshold=140,
                    )

                    variant_path = (
                        style_info["images_variant"]
                        / f"{base_name}_var_{var_idx:02d}.png"
                    )
                    variant_img.save(variant_path, optimize=True)

                    row = {
                        "filename": str(variant_path.relative_to(style_info["directory"])),
                        "style": style,
                        "kind": "variant",
                        "puzzle": puzzle_flat,
                        "solution": solution_flat,
                        "difficulty": difficulty,
                        "requested_blanks": blanks,
                        "actual_blanks": actual_blanks,
                        "seed": seed,
                    }
                    append_label(writer, labels_file, row)
                    total_written_by_style[style] += 1

            print(
                f"Generated puzzle {idx + 1}/{count} | "
                f"en={total_written_by_style['en']} | "
                f"fa={total_written_by_style['fa']}"
            )

    finally:
        for labels_file, _writer in label_resources.values():
            labels_file.close()

        metadata = {
            "seed": seed,
            "difficulty": difficulty,
            "requested_blanks": blanks,
            "count_per_language": count,
            "languages": list(styles),
            "variants_per_image": variants_per_image,
            "variant_severity": variant_severity,
            "max_variant_attempts": max_variant_attempts,
            "border_check": border_check,
            "english_font_filter": "latin_family_allowlist_plus_farsi_exclusions",
            "variant_blur_mode": "blur_heavy_noise_light",
            "font_dir": font_dir,
            "max_fonts_per_language": max_fonts_per_language,
            "selected_fonts": {
                style: [Path(font_path).name for font_path in font_pools[style]]
                for style in styles
            },
            "english_font_strategy": "curated_famous_english_fonts_only",
            "persian_font_strategy": "random_from_accepted_fonts",
            "output_directory": str(out_dir),
            "language_outputs": {
                style: {
                    "directory": str(style_outputs[style]["directory"]),
                    "original_images_directory": str(
                        style_outputs[style]["images_original"]
                    ),
                    "variant_images_directory": str(
                        style_outputs[style]["images_variant"]
                    ),
                    "labels_file": str(style_outputs[style]["labels"]),
                    "total_labeled_images_written": total_written_by_style[style],
                }
                for style in styles
            },
            "total_labeled_images_written": sum(total_written_by_style.values()),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    print()
    print("Done.")
    print(f"Output directory: {out_dir}")
    for style in styles:
        print(f"[{style}] directory: {style_outputs[style]['directory']}")
        print(
            f"[{style}] original images: "
            f"{style_outputs[style]['images_original']}"
        )
        print(
            f"[{style}] variant images: "
            f"{style_outputs[style]['images_variant']}"
        )
        print(f"[{style}] labels: {style_outputs[style]['labels']}")
        print(
            f"[{style}] total labeled images: "
            f"{total_written_by_style[style]}"
        )
    print(f"Metadata file: {metadata_path}")
    print(f"Seed: {seed}")
    print(f"Difficulty: {difficulty}")
    print(f"Requested blanks: {blanks}")
    print(f"Max fonts per language: {max_fonts_per_language}")
    print(f"Variant severity: {variant_severity}")
    print(f"Border check: {border_check}")
    print(
        "Total labeled images written: "
        f"{sum(total_written_by_style.values())}"
    )



# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate valid Sudoku image dataset with Persian and English digits."
    )

    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of Sudoku puzzles to generate. Default: 50",
    )

    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard", "expert"],
        help="Puzzle difficulty level.",
    )

    parser.add_argument(
        "--blanks",
        type=int,
        default=None,
        help="Custom number of empty cells. If set, overrides difficulty.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If not provided, a new random seed is generated.",
    )

    parser.add_argument(
        "--variants-per-image",
        type=int,
        default=3,
        help="Number of variant images per original image.",
    )

    parser.add_argument(
        "--out-root",
        type=str,
        default="sudoku_runs",
        help="Root directory for generated datasets.",
    )

    parser.add_argument(
        "--font-dir",
        type=str,
        default="Fonts",
        help="Directory containing fonts. Default: Fonts",
    )

    parser.add_argument(
        "--max-fonts-per-language",
        type=int,
        default=10,
        help="Maximum number of valid fonts used for each language. Default: 10",
    )

    parser.add_argument(
        "--variant-severity",
        type=float,
        default=1.2,
        help="Strength of image augmentations. Suggested: 0.9 to 1.4. Default: 1.2",
    )

    parser.add_argument(
        "--max-variant-attempts",
        type=int,
        default=12,
        help="How many times to retry generating a safe variant before fallback. Default: 12",
    )

    parser.add_argument(
        "--border-check",
        type=int,
        default=18,
        help="Minimum safe white border in pixels around dark content. Default: 18",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_dataset(
        out_root=args.out_root,
        count=args.count,
        difficulty=args.difficulty,
        blanks=args.blanks,
        variants_per_image=args.variants_per_image,
        font_dir=args.font_dir,
        max_fonts_per_language=args.max_fonts_per_language,
        seed=args.seed,
        variant_severity=args.variant_severity,
        max_variant_attempts=args.max_variant_attempts,
        border_check=args.border_check,
    )