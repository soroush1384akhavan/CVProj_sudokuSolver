# Sudoku Vision Solver

A computer vision–focused Sudoku solving system that extracts a Sudoku board from an input image, recognizes its digits, solves the puzzle, and projects the solution back onto the original image.

The project was developed as a **Computer Vision course project**, with the main focus on building and visualizing a complete image-processing pipeline using **OpenCV** and **PyTorch**.

A lightweight **FastAPI backend** and **React frontend** are included to make the computer vision pipeline interactive and easy to inspect.

---

## Computer Vision Pipeline

The core of this project is a multi-stage computer vision pipeline:

```text
Input Image
    ↓
Image Preprocessing
    ↓
Sudoku Grid Detection
    ↓
Perspective Correction
    ↓
81-Cell Extraction
    ↓
Digit Recognition
    ↓
Sudoku Solving
    ↓
Solution Overlay
```

Each stage is implemented separately so intermediate results can be inspected and debugged independently.

### 1. Image Preprocessing

The input Sudoku image is prepared for grid detection using OpenCV-based image processing.

The pipeline generates intermediate representations such as:

- Grayscale image
- Thresholded / binary image
- Processed image used for contour detection

This step reduces irrelevant visual information and makes the Sudoku grid easier to isolate.

---

### 2. Sudoku Grid Detection

The system searches the processed image for the Sudoku board.

The grid detection stage uses image geometry and contour analysis to locate the main Sudoku region.

After detecting the board, the pipeline identifies the grid boundaries and prepares the image for perspective correction.

---

### 3. Perspective Transformation

Sudoku images may be captured from an angle rather than perfectly from above.

To normalize the board, the detected Sudoku region is transformed into a top-down square representation.

```text
Camera Image
     ↓
Detected Grid
     ↓
Perspective Transform
     ↓
Normalized Sudoku Board
```

This produces a consistent board geometry for the following stages.

---

### 4. Cell Extraction

The normalized Sudoku board is divided into a **9 × 9 grid**, producing **81 individual cell images**.

Each cell is processed independently before digit recognition.

This separation is important because the recognition model receives individual Sudoku cells rather than the complete board image.

The application can also display a montage of the extracted cells for debugging and visual inspection.

---

### 5. Digit Recognition with PyTorch

Each extracted cell is passed through the digit-recognition stage.

A **PyTorch-based CNN pipeline** is provided for classifying Sudoku digits.

The recognition stage produces:

- Predicted digit
- Recognition confidence
- Empty-cell detection
- Low-confidence cell information

Low-confidence predictions can be highlighted in the frontend so the detected board can be manually corrected before solving.

This makes the system more robust when working with imperfect images or uncertain model predictions.

The digit-recognition implementation is located in:

```text
backend/phases/phase_04_digit_recognition/
```

Model files are stored separately under:

```text
backend/models/
```

The computer vision pipeline can still be inspected even when a trained digit model is not available. In that case, the detected board can be corrected manually before being sent to the solver.

---

### 6. Sudoku Solver

After the visual pipeline converts the image into a numerical 9 × 9 board, the extracted puzzle is passed to the Sudoku solver.

The solving stage operates on the recognized board rather than directly on the image.

This separation keeps the project divided into two clear problems:

```text
Computer Vision
Image → Sudoku Matrix

Algorithmic Solver
Sudoku Matrix → Solved Matrix
```

Users can manually correct recognition errors before solving.

---

### 7. Solution Overlay

Once the Sudoku puzzle is solved, the generated solution can be rendered back onto the processed board.

The overlay stage connects the algorithmic result back to the original visual input, completing the end-to-end pipeline:

```text
Original Image
      ↓
Visual Detection
      ↓
Digit Recognition
      ↓
Sudoku Solution
      ↓
Visual Overlay
```

---

## Pipeline Visualization

One of the main goals of the project is to make the computer vision process observable.

After processing an image, the application can display intermediate outputs including:

1. Original image
2. Grayscale image
3. Thresholded image
4. Detected Sudoku contour
5. Perspective-corrected board
6. Extracted 81-cell representation
7. Recognized Sudoku grid
8. Solved board / solution overlay

This is useful for understanding where errors occur in the pipeline instead of treating the vision system as a black box.

---

## Project Structure

```text
CVProj_sudokuSolver/
│
├── backend/
│   ├── app/
│   ├── common/
│   ├── pipeline/
│   │
│   ├── phases/
│   │   ├── phase_01_preprocessing/
│   │   ├── phase_02_grid_detection/
│   │   ├── phase_03_cell_extraction/
│   │   ├── phase_04_digit_recognition/
│   │   ├── phase_05_solver/
│   │   └── phase_06_overlay/
│   │
│   ├── models/
│   ├── samples/
│   ├── tests/
│   ├── requirements.txt
│   └── requirements-torch-cpu.txt
│
├── frontend/
│
├── docs/
│
├── README.md
└── RUN_WINDOWS.md
```

The phase-based backend structure reflects the actual computer vision workflow and allows each stage to be developed and tested independently.

---

## Technologies

### Computer Vision & Machine Learning

- Python
- OpenCV
- NumPy
- PyTorch
- Image preprocessing
- Contour detection
- Perspective transformation
- Image segmentation
- CNN-based digit recognition

### Backend

- FastAPI
- Uvicorn

### Frontend

- React
- TypeScript

The frontend is mainly used as an interactive visualization layer for uploading Sudoku images, inspecting CV stages, correcting detected cells, and displaying the final solution.

---

## Features

- Sudoku detection from uploaded images
- Image preprocessing and thresholding
- Grid contour detection
- Perspective correction
- Automatic 81-cell extraction
- PyTorch digit-recognition pipeline
- Confidence-aware predictions
- Manual correction of recognized cells
- Sudoku solving
- Solution overlay on the processed image
- Visualization of intermediate CV stages
- REST API for image processing and solving
- Interactive React interface

---

## Running the Project

### Backend

```bash
cd backend
pip install -r requirements.txt
```

For PyTorch CPU support:

```bash
pip install -r requirements-torch-cpu.txt
```

Start the backend:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

---

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

---

## API

### `POST /api/predict`

Uploads a Sudoku image and executes the computer vision pipeline.

The pipeline processes the image through grid detection, cell extraction, and digit recognition.

### `POST /api/solve`

Solves the detected or manually corrected Sudoku board and can generate the final visual overlay.

### `GET /health`

Backend health check.

---

## Why This Project

The main challenge of this project is not the Sudoku solving algorithm itself.

The more interesting problem is transforming a real image into a reliable structured Sudoku board.

That requires combining several computer vision tasks:

- preprocessing noisy images
- detecting geometric structures
- correcting perspective distortion
- segmenting the board into cells
- distinguishing empty cells from digits
- classifying extracted digits
- handling uncertain predictions
- mapping the final solution back into image space

The project therefore demonstrates an end-to-end computer vision workflow rather than only an implementation of a Sudoku solver.

---

## Future Improvements

Possible improvements include:

- Training the digit classifier on a larger real-world Sudoku dataset
- Evaluating recognition accuracy on a dedicated test set
- Adding a confusion matrix for digit classification
- Improving grid detection under difficult lighting conditions
- Improving robustness to blur and extreme perspective distortion
- Benchmarking each CV stage independently
- Supporting live camera input
- Improving automatic handling of low-confidence predictions

---

## Course Project

Developed as a **Computer Vision course project** with emphasis on:

- classical image processing
- geometric computer vision
- image segmentation
- deep-learning-based recognition
- end-to-end vision pipeline design
