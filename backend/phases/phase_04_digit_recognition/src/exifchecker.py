from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[3]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import cv2

from common.images import imread_color


image_path = (
    BACKEND_ROOT
    / "storage"
    / "sudoku"
    / "raw"
    / "v2_train"
    / "v2_train"
    / "image1040.jpg"
)

output_path = BACKEND_ROOT / "image65_fixed.png"

image = imread_color(image_path)

# خروجی فعلی را ۹۰ درجه پادساعت‌گرد می‌چرخانیم
# image = cv2.rotate(
#     image,
#     cv2.ROTATE_90_COUNTERCLOCKWISE,
# )

saved = cv2.imwrite(str(output_path), image)

print("Saved:", saved)
print("Output:", output_path)