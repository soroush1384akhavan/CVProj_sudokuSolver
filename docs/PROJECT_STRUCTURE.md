# Project Structure

```text
sudoku-cv-windows-pytorch-opencv/
├── backend/
│   ├── app/                         # FastAPI app
│   ├── common/                      # shared file/image helpers
│   ├── phases/
│   │   ├── phase_01_preprocessing/
│   │   ├── phase_02_grid_detection/
│   │   ├── phase_03_cell_extraction/
│   │   ├── phase_04_digit_recognition_pytorch/
│   │   ├── phase_05_solver/
│   │   └── phase_06_overlay/
│   ├── pipeline/                    # connects phases together
│   ├── models/                      # digit_cnn.pth goes here
│   ├── samples/                     # sample Sudoku image
│   └── tests/
└── frontend/
    ├── src/api/
    ├── src/components/
    ├── src/hooks/
    └── src/styles.css
```
