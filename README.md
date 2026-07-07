# Sudoku CV Windows - PyTorch + OpenCV

A clean Windows-friendly Sudoku computer vision project with a React frontend similar to the reference Sudoku Solver UI flow:

- Upload Sudoku image
- Preview image
- Process image
- View detected 9×9 board
- Edit/correct cells manually
- Highlight low-confidence cells
- Solve puzzle
- Reset / Clear
- View every computer-vision phase result in the UI

The backend is divided by project phase:

```text
backend/phases/
├── phase_01_preprocessing/
├── phase_02_grid_detection/
├── phase_03_cell_extraction/
├── phase_04_digit_recognition_pytorch/
├── phase_05_solver/
└── phase_06_overlay/
```

## Important

The backend works even before you train a digit model. If `backend/models/digit_cnn.pth` is missing, the image pipeline still runs and shows debug images, but digit recognition returns mostly empty cells. You can still edit the board manually in the UI and solve it.

## Windows quick run

### 1) Backend

```powershell
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If you want PyTorch CPU-only:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2) Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Backend docs:

```text
http://127.0.0.1:8000/docs
```

## API

- `POST /api/predict` — image upload and full CV pipeline
- `POST /api/solve` — solve edited board and optionally create overlay
- `GET /health` — backend health

## Frontend debug phases

After uploading an image and clicking **Process Image**, the right panel shows:

1. Original uploaded image
2. Grayscale image
3. Threshold image
4. Detected grid contour
5. Warped Sudoku board
6. Extracted 81-cell montage
7. Solved warped board / original overlay after solving

## Model training placeholder

The PyTorch model structure is in:

```text
backend/phases/phase_04_digit_recognition_pytorch/model.py
```

A clean training entry point is provided:

```text
backend/phases/phase_04_digit_recognition_pytorch/train_digit_cnn.py
```

You can train on MNIST/synthetic/real cell crops and save:

```text
backend/models/digit_cnn.pth
```

## Recommended workflow

1. First run backend and frontend.
2. Upload a Sudoku image and inspect phase outputs.
3. Manually correct the grid and solve.
4. Train/fine-tune the digit CNN later.
5. Add real evaluation metrics and confusion matrix for your report.
