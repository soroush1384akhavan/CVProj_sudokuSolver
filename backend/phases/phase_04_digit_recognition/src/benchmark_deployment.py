# Run examples:
# python -m phases.phase_04_digit_recognition.src.benchmark_deployment --language en
# python -m phases.phase_04_digit_recognition.src.benchmark_deployment --language fa

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from app.config import settings
from .dataset import get_phase4_config
from .model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export and benchmark PyTorch, TorchScript and ONNX models on CPU."
    )
    parser.add_argument("--language", choices=["en", "fa"], default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-dir", default="deployment_benchmarks")
    return parser.parse_args()


def resolve_language(cli_language: str | None) -> str:
    language = cli_language or settings.get("digit_recognition.default_language", "en")
    language = str(language).strip().lower()

    if language not in ("en", "fa"):
        raise ValueError(f"Unsupported language: {language}")

    return language


def resolve_checkpoint_path(language: str) -> Path:
    configured_path = settings.get(f"digit_recognition.model_paths.{language}", None)

    if not configured_path:
        raise KeyError(
            f"Missing config: digit_recognition.model_paths.{language}"
        )

    checkpoint_path = settings.resolve_path(str(configured_path))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model not found for language={language}: {checkpoint_path}"
        )

    return checkpoint_path


def load_model(checkpoint_path: Path, num_classes: int, dropout: float) -> torch.nn.Module:
    model = build_model(num_classes=num_classes, dropout=dropout).cpu()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_torchscript(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: Path,
) -> torch.jit.ScriptModule:
    with torch.inference_mode():
        traced_model = torch.jit.trace(model, example_input)
        traced_model = torch.jit.freeze(traced_model.eval())

    traced_model.save(str(output_path))
    return traced_model


def export_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install ONNX with: pip install onnx") from exc

    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def build_onnx_session(onnx_path: Path, cpu_threads: int):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Install ONNX Runtime with: pip install onnxruntime"
        ) from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def benchmark_forward(
    forward_fn: Callable[[], object],
    warmup: int,
    runs: int,
) -> np.ndarray:
    for _ in range(warmup):
        forward_fn()

    timings_ms = np.empty(runs, dtype=np.float64)

    for index in range(runs):
        started = time.perf_counter_ns()
        forward_fn()
        timings_ms[index] = (time.perf_counter_ns() - started) / 1_000_000.0

    return timings_ms


def summarize(
    model_format: str,
    model_path: Path,
    timings_ms: np.ndarray,
    batch_size: int,
    max_output_diff: float,
) -> dict[str, float | str]:
    average_ms = float(timings_ms.mean())

    return {
        "format": model_format,
        "model_file": model_path.name,
        "size_mb": model_path.stat().st_size / (1024 ** 2),
        "avg_latency_ms": average_ms,
        "median_latency_ms": float(np.median(timings_ms)),
        "p95_latency_ms": float(np.percentile(timings_ms, 95)),
        "min_latency_ms": float(timings_ms.min()),
        "max_latency_ms": float(timings_ms.max()),
        "std_latency_ms": float(timings_ms.std()),
        "throughput_samples_s": batch_size * 1000.0 / average_ms,
        "max_output_diff": max_output_diff,
    }


def add_comparison_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    pytorch_row = results_df.loc[results_df["format"] == "PyTorch"].iloc[0]
    base_latency = float(pytorch_row["avg_latency_ms"])
    base_size = float(pytorch_row["size_mb"])

    results_df["speedup_vs_pytorch"] = base_latency / results_df["avg_latency_ms"]
    results_df["latency_reduction_pct"] = (
        (base_latency - results_df["avg_latency_ms"]) / base_latency * 100.0
    )
    results_df["size_reduction_pct"] = (
        (base_size - results_df["size_mb"]) / base_size * 100.0
    )

    return results_df


def print_clean_table(results_df: pd.DataFrame) -> None:
    display_df = results_df[
        [
            "format",
            "size_mb",
            "avg_latency_ms",
            "median_latency_ms",
            "p95_latency_ms",
            "throughput_samples_s",
            "speedup_vs_pytorch",
            "latency_reduction_pct",
            "size_reduction_pct",
            "max_output_diff",
        ]
    ].copy()

    display_df.columns = [
        "Format",
        "Size (MB)",
        "Avg Latency (ms)",
        "Median (ms)",
        "P95 (ms)",
        "Throughput (sample/s)",
        "Speedup",
        "Latency Reduction (%)",
        "Size Reduction (%)",
        "Max Output Diff",
    ]

    formatters = {
        "Size (MB)": lambda value: f"{value:.3f}",
        "Avg Latency (ms)": lambda value: f"{value:.4f}",
        "Median (ms)": lambda value: f"{value:.4f}",
        "P95 (ms)": lambda value: f"{value:.4f}",
        "Throughput (sample/s)": lambda value: f"{value:.2f}",
        "Speedup": lambda value: f"{value:.2f}x",
        "Latency Reduction (%)": lambda value: f"{value:+.2f}",
        "Size Reduction (%)": lambda value: f"{value:+.2f}",
        "Max Output Diff": lambda value: f"{value:.8f}",
    }

    print("\n" + "=" * 148)
    print("CPU DEPLOYMENT BENCHMARK")
    print("=" * 148)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.colheader_justify", "center",
    ):
        print(display_df.to_string(index=False, formatters=formatters))


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")
    if args.runs < 1:
        raise ValueError("runs must be at least 1")
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be at least 1")

    language = resolve_language(args.language)
    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})

    image_size = int(model_cfg.get("image_size", 28))
    num_classes = int(model_cfg.get("num_classes", 10))
    dropout = float(model_cfg.get("dropout", 0.5))

    torch.set_num_threads(args.cpu_threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    checkpoint_path = resolve_checkpoint_path(language)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / language / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    torchscript_path = output_dir / "digit_cnn_torchscript.pt"
    onnx_path = output_dir / "digit_cnn.onnx"

    print(f"Language: {language}")
    print(f"CPU threads: {args.cpu_threads}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warm-up runs: {args.warmup}")
    print(f"Measured runs: {args.runs}")
    print(f"PyTorch checkpoint: {checkpoint_path.resolve()}")

    model = load_model(checkpoint_path, num_classes, dropout)

    example_input = torch.rand(
        args.batch_size,
        1,
        image_size,
        image_size,
        dtype=torch.float32,
    )

    torchscript_model = export_torchscript(
        model,
        example_input,
        torchscript_path,
    )

    export_onnx(
        model,
        example_input,
        onnx_path,
        args.opset,
    )

    onnx_session = build_onnx_session(
        onnx_path,
        args.cpu_threads,
    )

    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_input = example_input.detach().cpu().numpy()

    with torch.inference_mode():
        pytorch_reference = model(example_input).detach().cpu()
        torchscript_reference = torchscript_model(example_input).detach().cpu()

    onnx_reference = onnx_session.run(
        None,
        {onnx_input_name: onnx_input},
    )[0]

    torchscript_diff = float(
        torch.max(torch.abs(pytorch_reference - torchscript_reference)).item()
    )
    onnx_diff = float(
        np.max(np.abs(pytorch_reference.numpy() - onnx_reference))
    )

    with torch.inference_mode():
        pytorch_timings = benchmark_forward(
            lambda: model(example_input),
            args.warmup,
            args.runs,
        )
        torchscript_timings = benchmark_forward(
            lambda: torchscript_model(example_input),
            args.warmup,
            args.runs,
        )

    onnx_timings = benchmark_forward(
        lambda: onnx_session.run(None, {onnx_input_name: onnx_input}),
        args.warmup,
        args.runs,
    )

    results_df = pd.DataFrame(
        [
            summarize(
                "PyTorch",
                checkpoint_path,
                pytorch_timings,
                args.batch_size,
                0.0,
            ),
            summarize(
                "TorchScript",
                torchscript_path,
                torchscript_timings,
                args.batch_size,
                torchscript_diff,
            ),
            summarize(
                "ONNX",
                onnx_path,
                onnx_timings,
                args.batch_size,
                onnx_diff,
            ),
        ]
    )

    results_df = add_comparison_columns(results_df)
    print_clean_table(results_df)

    csv_path = output_dir / "deployment_benchmark.csv"
    json_path = output_dir / "deployment_benchmark.json"

    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report = {
        "language": language,
        "checkpoint": str(checkpoint_path.resolve()),
        "image_size": image_size,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "cpu_threads": args.cpu_threads,
        "opset": args.opset,
        "results": results_df.to_dict(orient="records"),
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    print(f"TorchScript: {torchscript_path.resolve()}")
    print(f"ONNX:        {onnx_path.resolve()}")
    print(f"CSV:         {csv_path.resolve()}")
    print(f"JSON:        {json_path.resolve()}")


if __name__ == "__main__":
    main()