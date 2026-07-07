from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import build_digit_dataloaders, build_eval_only_datasets, get_phase4_config
from .model import build_model
from .utils import new_run_dir, save_json, set_seed, save_training_plots


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False, desc="train"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False, desc="val"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_per_dataset(model, datasets_dict: dict, device, batch_size: int = 64) -> dict:
    model.eval()
    results = {}

    for name, dataset in datasets_dict.items():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        correct, total = 0, 0

        for images, labels in tqdm(loader, leave=False, desc=f"eval:{name}"):
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        results[name] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "num_samples": total,
        }

    return results


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
    save_path,
    early_stopping_patience: int = 6,
    lr_scheduler_patience: int = 3,
    lr_scheduler_factor: float = 0.5,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )
    criterion = nn.CrossEntropyLoss()

    history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "lr": [],
        }
    best_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_accuracy)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_accuracy={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | val_accuracy={val_accuracy:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved (accuracy={val_accuracy:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    return history


def main() -> None:
    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})
    train_cfg = phase_cfg.get("training", {})
    data_cfg = phase_cfg.get("data", {})
    aug_cfg = phase_cfg.get("augmentation", {})

    seed = set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_dir = new_run_dir()
    checkpoint_name = str(train_cfg.get("checkpoint_name", "digit_cnn.pth"))
    checkpoint_path = run_dir / checkpoint_name
    print(f"Run directory: {run_dir}")

    train_loader, val_loader, test_loader = build_digit_dataloaders()
    per_dataset_eval_sets = build_eval_only_datasets()

    num_classes = int(model_cfg.get("num_classes", 10))
    dropout = float(model_cfg.get("dropout", 0.5))
    epochs = int(train_cfg.get("epochs", 40))
    learning_rate = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0005))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 6))
    lr_scheduler_patience = int(train_cfg.get("lr_scheduler_patience", 3))
    lr_scheduler_factor = float(train_cfg.get("lr_scheduler_factor", 0.5))

    model = build_model(num_classes=num_classes, dropout=dropout).to(device)

    used_config = {
        "model": model_cfg,
        "training": train_cfg,
        "data": data_cfg,
        "augmentation": aug_cfg,
        "seed": seed,
        "device": str(device),
        "model_class": model.__class__.__name__,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    save_json(run_dir / "config_used.json", used_config)

    print(f"\nModel: {model.__class__.__name__} | num_classes={num_classes}")
    print(f"Total params: {used_config['total_params']:,}")
    print("Starting training...\n")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        save_path=checkpoint_path,
        early_stopping_patience=early_stopping_patience,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
    )
    
    plot_paths = save_training_plots(history, run_dir)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_accuracy = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"\nOverall test evaluation: loss={test_loss:.4f} | accuracy={test_accuracy:.4f}")

    print("\nPer-dataset test accuracy:")
    per_dataset_results = evaluate_per_dataset(model, per_dataset_eval_sets, device)
    for name, result in per_dataset_results.items():
        print(f"  {name:12s} -> accuracy={result['accuracy']:.4f} (n={result['num_samples']})")

    report = {
        "run_dir": str(run_dir),
        "seed": seed,
        "device": str(device),
        "num_classes": num_classes,
        "epochs_ran": len(history["train_loss"]),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "history": history,
        "overall_test_loss": test_loss,
        "overall_test_accuracy": test_accuracy,
        "per_dataset_accuracy": per_dataset_results,
        "checkpoint_path": str(checkpoint_path),
        "plot_paths": plot_paths,
    }
    save_json(run_dir / "training_report.json", report)

    print(f"\nTraining complete.")
    print(f"All outputs saved to: {run_dir}")
    print(f"  - config_used.json     ")
    print(f"  - training_report.json ")
    print(f"  - digit_cnn.pth  ")
    print(f"  - loss_curve.png       ")
    print(f"  - accuracy_curve.png")
    print(f"  - lr_curve.png         (")


if __name__ == "__main__":
    main()
