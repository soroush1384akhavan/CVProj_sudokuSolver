# Sudoku Dataset Generator

این فایل راهنمای اجرای اسکریپت `generate_sudoku_dataset.py` است.

اسکریپت برای ساخت دیتاست تصویر سودوکو استفاده می‌شود و خروجی شامل تصویرهای اصلی، تصویرهای تغییر‌داده‌شده، فایل لیبل‌ها و متادیتا است.

---

## 1. نصب پیش‌نیازها

قبل از اجرا، این پکیج‌ها را نصب کن:

```bash
pip install pillow numpy
```

---

## 2. ساختار پیشنهادی پروژه

بهتر است فایل‌ها این‌طوری کنار هم باشند:

```text
cvProj/
├── generate_sudoku_dataset.py
├── README.md
├── Fonts/
│   ├── Vazirmatn-Regular.ttf
│   ├── Vazirmatn-Bold.ttf
│   └── ...
└── sudoku_runs/
```

پوشه `Fonts` برای فونت‌های فارسی و انگلیسی استفاده می‌شود. کد به‌صورت رندوم از فونت‌های سالم داخل این پوشه استفاده می‌کند.

---

## 3. اجرای ساده

```bash
python generate_sudoku_dataset.py
```

در حالت پیش‌فرض:

```text
count = 50
difficulty = medium
variants_per_image = 3
font_dir = Fonts
out_root = sudoku_runs
```

---

## 4. دیدن راهنمای برنامه

```bash
python generate_sudoku_dataset.py --help
```

---

## 5. پارامترهای قابل استفاده

### تعداد سودوکوها

```bash
python generate_sudoku_dataset.py --count 50
```

مثلاً:

```bash
python generate_sudoku_dataset.py --count 10
python generate_sudoku_dataset.py --count 100
```

---

### سختی سودوکو

```bash
python generate_sudoku_dataset.py --difficulty medium
```

گزینه‌ها:

```text
easy    = حدود 35 خانه خالی
medium  = حدود 45 خانه خالی
hard    = حدود 55 خانه خالی
expert  = حدود 60 خانه خالی
```

مثال:

```bash
python generate_sudoku_dataset.py --difficulty easy
python generate_sudoku_dataset.py --difficulty medium
python generate_sudoku_dataset.py --difficulty hard
python generate_sudoku_dataset.py --difficulty expert
```

---

### تعداد خانه‌های خالی سفارشی

اگر `--blanks` بدهی، از `--difficulty` مهم‌تر است.

```bash
python generate_sudoku_dataset.py --blanks 50
```

مثال:

```bash
python generate_sudoku_dataset.py --difficulty easy --blanks 55
```

در این مثال، با اینکه difficulty برابر easy است، تعداد خانه‌های خالی 55 می‌شود.

---

### Seed

برای اینکه خروجی قابل تکرار باشد:

```bash
python generate_sudoku_dataset.py --seed 42
```

مثال:

```bash
python generate_sudoku_dataset.py --seed 123
python generate_sudoku_dataset.py --seed 777
```

اگر seed ندهی، هر بار خروجی جدید ساخته می‌شود:

```bash
python generate_sudoku_dataset.py
```

---

### تعداد تصویرهای تغییر‌داده‌شده برای هر تصویر اصلی

```bash
python generate_sudoku_dataset.py --variants-per-image 5
```

مثال:

```bash
python generate_sudoku_dataset.py --variants-per-image 1
python generate_sudoku_dataset.py --variants-per-image 3
python generate_sudoku_dataset.py --variants-per-image 8
```

برای هر سودوکو دو تصویر اصلی ساخته می‌شود:

```text
1 تصویر انگلیسی
1 تصویر فارسی
```

و برای هرکدام به تعداد `variants_per_image` تصویر تغییر‌داده‌شده ساخته می‌شود.

مثلاً اگر:

```text
count = 50
variants_per_image = 3
```

تعداد کل تصاویر:

```text
50 × 2 × (1 + 3) = 400 تصویر
```

---

### پوشه خروجی

```bash
python generate_sudoku_dataset.py --out-root sudoku_runs
```

مثال:

```bash
python generate_sudoku_dataset.py --out-root outputs
python generate_sudoku_dataset.py --out-root dataset_runs
```

در ویندوز، اگر مسیر فاصله داشت، داخل کوتیشن بگذار:

```bash
python generate_sudoku_dataset.py --out-root "D:\my projects\sudoku_outputs"
```

---

### پوشه فونت‌ها

حالت پیش‌فرض:

```bash
python generate_sudoku_dataset.py --font-dir Fonts
```

در PowerShell ویندوز:

```powershell
python generate_sudoku_dataset.py --font-dir .\Fonts
```

اگر فونت‌ها جای دیگری هستند:

```bash
python generate_sudoku_dataset.py --font-dir "D:\git1\cv\cvProj\Fonts"
```

---

## 6. کامندهای پیشنهادی

### تست سریع

```bash
python generate_sudoku_dataset.py --count 5 --difficulty medium --seed 42
```

### دیتاست معمولی پیشنهادی

```bash
python generate_sudoku_dataset.py --count 50 --difficulty medium --variants-per-image 3 --seed 42 --font-dir Fonts
```

### دیتاست سخت‌تر

```bash
python generate_sudoku_dataset.py --count 50 --difficulty hard --variants-per-image 5 --seed 42 --font-dir Fonts
```

### دیتاست با خروجی جدید در هر اجرا

```bash
python generate_sudoku_dataset.py --count 50 --difficulty medium --variants-per-image 3 --font-dir Fonts
```

در این حالت seed ندادی، پس هر بار خروجی جدید ساخته می‌شود.

### دیتاست با خانه‌های خالی سفارشی

```bash
python generate_sudoku_dataset.py --count 50 --blanks 52 --variants-per-image 5 --seed 123 --font-dir Fonts
```

### دیتاست کامل با همه ورودی‌ها

```bash
python generate_sudoku_dataset.py --count 50 --difficulty hard --blanks 55 --seed 42 --variants-per-image 5 --out-root sudoku_runs --font-dir Fonts
```

---

## 7. خروجی برنامه

بعد از اجرا، داخل پوشه `sudoku_runs` یک پوشه جدید ساخته می‌شود. مثلاً:

```text
sudoku_runs/
└── sudoku_hard_20260706_153000_123456_seed_42/
    ├── images_original/
    │   ├── sudoku_00000_en.png
    │   ├── sudoku_00000_fa.png
    │   └── ...
    ├── images_variant/
    │   ├── sudoku_00000_en_var_00.png
    │   ├── sudoku_00000_en_var_01.png
    │   ├── sudoku_00000_fa_var_00.png
    │   └── ...
    ├── labels.csv
    └── metadata.json
```

---

## 8. معنی پوشه‌ها

### `images_original`

تصویرهای تمیز و اصلی سودوکوها.

```text
images_original/sudoku_00000_en.png
images_original/sudoku_00000_fa.png
```

### `images_variant`

تصویرهای تغییر‌داده‌شده برای آموزش مدل.

این تصاویر شامل تغییراتی مثل موارد زیر هستند:

```text
noise
perspective / projection
rotation
shear
shadow
uneven lighting
blur
motion blur
jpeg artifact
resolution degradation
paper texture
```

---

## 9. فایل labels.csv

جواب درست همه تصاویر داخل فایل زیر است:

```text
labels.csv
```

ستون‌های مهم:

```text
filename          مسیر تصویر
style             en یا fa
kind              original یا variant
puzzle            جدول ناقص، خانه‌های خالی با 0
solution          جواب کامل درست سودوکو
difficulty        سطح سختی
requested_blanks  تعداد خانه خالی درخواستی
actual_blanks     تعداد خانه خالی واقعی
seed              seed اجرا
```

---

## 10. جواب درست کجاست؟

جواب درست هر تصویر داخل ستون زیر است:

```text
solution
```

این مقدار یک رشته 81 رقمی است.

مثلاً:

```text
534678912672195348198342567859761423426853791713924856961537284287419635345286179
```

یعنی:

```text
534678912
672195348
198342567
859761423
426853791
713924856
961537284
287419635
345286179
```

---

## 11. جدول ناقص کجاست؟

جدول ناقص داخل ستون زیر است:

```text
puzzle
```

خانه‌های خالی با `0` مشخص می‌شوند.

---

## 12. اجرای مستقیم از داخل یک فایل Python دیگر

```python
from generate_sudoku_dataset import generate_dataset

generate_dataset(
    out_root="sudoku_runs",
    count=50,
    difficulty="hard",
    blanks=None,
    variants_per_image=5,
    font_dir="Fonts",
    seed=42,
)
```

با تعداد خانه خالی سفارشی:

```python
from generate_sudoku_dataset import generate_dataset

generate_dataset(
    out_root="sudoku_runs",
    count=50,
    difficulty="medium",
    blanks=52,
    variants_per_image=5,
    font_dir="Fonts",
    seed=123,
)
```

---

## 13. نکات مهم فونت

اگر تصویر فارسی به‌جای عدد مربع نشان داد، مشکل از فونت است. داخل پوشه `Fonts` فقط فونت‌های متنی سالم بگذار.

فونت‌های پیشنهادی:

```text
Vazirmatn
IRANSans
Tahoma
Arial
BNazanin
Shabnam
Sahel
Yekan
```

فونت‌های آیکونی را داخل `Fonts` نگذار:

```text
Wingdings
Webdings
FontAwesome
Material Icons
Emoji
Symbol
```

---

## 14. اگر برنامه وسط اجرا قطع شد

فایل `labels.csv` همزمان با ساخت تصاویر نوشته می‌شود. پس اگر برنامه وسط اجرا قطع شود، لیبل‌های ساخته‌شده تا همان لحظه داخل `labels.csv` باقی می‌مانند.

اما بهتر است run ناقص را برای آموزش استفاده نکنی، مگر اینکه مطمئن شوی فایل‌های داخل `labels.csv` واقعاً وجود دارند.

---

## 15. مهم‌ترین کامند پیشنهادی

```bash
python generate_sudoku_dataset.py --count 50 --difficulty hard --seed 42 --variants-per-image 5 --font-dir Fonts
```
