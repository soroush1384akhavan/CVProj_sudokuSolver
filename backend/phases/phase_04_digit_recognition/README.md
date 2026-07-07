# Phase 4 — PyTorch Digit Recognition

This phase is intentionally structured like a standalone practical ML project.

```text
phase_04_digit_recognition_pytorch/
├── data/
│   ├── raw/
│   ├── processed/
│   └── labeled_cells/
│       ├── 0/
│       ├── 1/
│       └── ... 9/
├── outputs/
│   ├── checkpoints/
│   ├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── model.py
│   ├── classifier.py
│   ├── train.py
│   └── utils.py
├── classifier.py       # compatibility wrapper used by the main pipeline
├── model.py            # compatibility wrapper
├── train_digit_cnn.py  # entry point wrapper
└── README.md
```

## One global YAML config

All important project constants are in:

```text
backend/config.yml
```

Examples:

```yaml
digit_recognition:
  model_path: models/digit_cnn.pth
  confidence_threshold: 0.75
  model:
    name: DigitCNN
    image_size: 28
    num_classes: 10
    dropout: 0.25
  training:
    batch_size: 64
    epochs: 20
  data:
    validation_split: 0.1
  augmentation:
    enabled: true
```

## Expected final model path

```text
backend/models/digit_cnn.pth
```

Class mapping:

```text
0 = empty
1 = digit 1
...
9 = digit 9
```

The backend works without a model. Without `digit_cnn.pth`, it still shows OpenCV phase outputs and lets the user manually edit the board.
