from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from app.config import settings

from .dataset import DigitTransform, get_phase4_config
from .datasets.fromsoduko.charls_soduku_dataloader import DatSudokuCellDataset
from .datasets.fromsoduko.generated_dataloader import GeneratedSudokuCellDataset
from .model import build_model
from .train import train_model, validate
from .utils import latest_checkpoint_path, new_run_dir, save_json, save_training_plots, set_seed


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    return [str(item) for item in value]


def _resolve_path(value: str | Path) -> Path:
    return settings.resolve_path(str(value))


# اصلاح اول: اضافه کردن فلگ کنترل آگمنتیشن به تابع ساخت دیتابیس
def _build_source_datasets(
    transform: DigitTransform,
    sudoku_cfg: dict[str, Any],
    apply_safe_augmentation: bool = False,  # اضافه شدن پارامتر جدید
) -> tuple[list, dict[str, int]]:
    cache_root = _resolve_path(sudoku_cfg.get("cache_dir", "storage/sudoku/cell_cache"))
    include_empty_cells = bool(sudoku_cfg.get("include_empty_cells", False))
    refresh_cache = bool(sudoku_cfg.get("refresh_cache", False))

    datasets = []
    summary: dict[str, int] = {}

    for root in _as_list(sudoku_cfg.get("generated_roots")):
        root_path = _resolve_path(root)
        dataset = GeneratedSudokuCellDataset(
            root_dir=root_path,
            cache_dir=cache_root / "generated" / root_path.name,
            include_empty_cells=include_empty_cells,
            transform=transform,
            refresh_cache=refresh_cache,
            apply_safe_augmentation=apply_safe_augmentation,  # پاس دادن فلگ به کلاس اول
        )
        datasets.append(dataset)
        summary[f"generated:{root_path.name}"] = len(dataset)

    for root in _as_list(sudoku_cfg.get("dat_roots")):
        root_path = _resolve_path(root)
        dataset = DatSudokuCellDataset(
            root_dir=root_path,
            cache_dir=cache_root / "dat" / root_path.name,
            include_empty_cells=include_empty_cells,
            transform=transform,
            refresh_cache=refresh_cache,
            apply_safe_augmentation=apply_safe_augmentation,  # پاس دادن فلگ به کلاس دوم
        )
        datasets.append(dataset)
        summary[f"dat:{root_path.name}"] = len(dataset)

    if not datasets:
        raise ValueError("No Sudoku fine-tuning datasets configured.")

    return datasets, summary


def build_sudoku_finetune_loaders():
    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})
    train_cfg = phase_cfg.get("training", {})
    data_cfg = phase_cfg.get("data", {})
    aug_cfg = phase_cfg.get("augmentation", {})
    sudoku_cfg = data_cfg.get("sudoku_cells", {})

    image_size = int(model_cfg.get("image_size", 28))
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 0))
    validation_split = float(sudoku_cfg.get("validation_split", 0.1))
    seed = int(train_cfg.get("seed", 42))
    augmentation_enabled = bool(aug_cfg.get("enabled", True))

    train_transform = DigitTransform(
        size=image_size,
        augment=augmentation_enabled,
        augment_config=aug_cfg,
    )
    eval_transform = DigitTransform(
        size=image_size,
        augment=False,
        augment_config=aug_cfg,
    )

    # اصلاح دوم: ساخت جداگانه منابع دیتابیس برای اینکه آگمنتیشن فقط به کدهای Train اعمال شود
    train_sources, summary = _build_source_datasets(
        train_transform, sudoku_cfg, apply_safe_augmentation=augmentation_enabled
    )
    eval_sources, _ = _build_source_datasets(
        eval_transform, sudoku_cfg, apply_safe_augmentation=False  # داده‌های ارزیابی هرگز نباید آگمنت شوند
    )

    train_full = ConcatDataset(train_sources)
    eval_full = ConcatDataset(eval_sources)

    n_total = len(train_full)

    if n_total < 2:
        raise ValueError("Sudoku fine-tuning dataset must contain at least 2 cells.")

    n_val = int(n_total * validation_split)
    n_val = max(1, min(n_total - 1, n_val))
    n_train = n_total - n_val

    indices = torch.randperm(
        n_total,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # اصلاح سوم: تخصیص صحیح Subsetها از دیتابیس‌های مربوط به خودشان
    train_loader = DataLoader(
        Subset(train_full, train_indices),  # استفاده از منبع شامل آگمنتیشن امن
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(eval_full, val_indices),    # استفاده از منبع تمیز و بدون تغییر برای تست واقعی
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, summary


def main() -> None:
    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})
    train_cfg = phase_cfg.get("training", {})
    fine_tune_cfg = phase_cfg.get("sudoku_fine_tune", {})
    data_cfg = phase_cfg.get("data", {})
    aug_cfg = phase_cfg.get("augmentation", {})

    seed = set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, source_summary = build_sudoku_finetune_loaders()

    num_classes = int(model_cfg.get("num_classes", 10))
    dropout = float(model_cfg.get("dropout", 0.5))
    model = build_model(num_classes=num_classes, dropout=dropout).to(device)

    checkpoint_override = fine_tune_cfg.get("checkpoint_path")
    checkpoint_path = (
        _resolve_path(checkpoint_override)
        if checkpoint_override
        else latest_checkpoint_path()
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    epochs = int(fine_tune_cfg.get("epochs", 5))
    learning_rate = float(fine_tune_cfg.get("learning_rate", 0.0001))
    weight_decay = float(fine_tune_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0001)))
    early_stopping_patience = int(fine_tune_cfg.get("early_stopping_patience", 3))
    lr_scheduler_patience = int(fine_tune_cfg.get("lr_scheduler_patience", 2))
    lr_scheduler_factor = float(fine_tune_cfg.get("lr_scheduler_factor", 0.5))

    run_dir = new_run_dir()
    checkpoint_name = str(train_cfg.get("checkpoint_name", "digit_cnn.pth"))
    output_checkpoint_path = run_dir / checkpoint_name

    used_config = {
        "model": model_cfg,
        "training": train_cfg,
        "sudoku_fine_tune": fine_tune_cfg,
        "data": data_cfg,
        "augmentation": aug_cfg,
        "source_checkpoint_path": str(checkpoint_path),
        "source_summary": source_summary,
        "seed": seed,
        "device": str(device),
    }
    save_json(run_dir / "config_used.json", used_config)

    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Run directory: {run_dir}")
    print("Sudoku fine-tune samples:")
    for name, count in source_summary.items():
        print(f"  {name}: {count}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        save_path=output_checkpoint_path,
        early_stopping_patience=early_stopping_patience,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
    )

    plot_paths = save_training_plots(history, run_dir)

    model.load_state_dict(torch.load(output_checkpoint_path, map_location=device, weights_only=True))
    val_loss, val_accuracy = validate(model, val_loader, nn.CrossEntropyLoss(), device)

    report = {
        "run_dir": str(run_dir),
        "source_checkpoint_path": str(checkpoint_path),
        "checkpoint_path": str(output_checkpoint_path),
        "seed": seed,
        "device": str(device),
        "epochs_ran": len(history["train_loss"]),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "source_summary": source_summary,
        "history": history,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "plot_paths": plot_paths,
    }
    save_json(run_dir / "training_report.json", report)

    print(f"\nSudoku fine-tuning complete: val_loss={val_loss:.4f} | val_accuracy={val_accuracy:.4f}")
    print(f"All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()