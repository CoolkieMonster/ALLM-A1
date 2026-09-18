"""Plot Task 2 train and validation bits-per-byte curves from JSONL logs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(path):
    config = {}
    metrics = {"train": [], "validation": []}
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if record.get("record") == "config":
                config = record
            elif record.get("record") == "metric":
                split = record.get("split")
                if split in metrics:
                    metrics[split].append(record)
    return config, metrics


def moving_average(values, window=10):
    smoothed = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        smoothed.append(running_sum / min(index + 1, window))
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Plot Task 2 BPB curves")
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    vocab_sizes = (8192, 32768)
    loaded = {}
    rows = []

    for vocab_size in vocab_sizes:
        metrics_path = results_root / f"vocab_{vocab_size}" / "base_train_bpb_metrics.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        config, metrics = load_run(metrics_path)
        loaded[vocab_size] = (config, metrics)
        for split, records in metrics.items():
            for record in records:
                rows.append({
                    "vocab_size": vocab_size,
                    "split": split,
                    "step": record["step"],
                    "tokens_seen": record["tokens_seen"],
                    "bpb": record["bpb"],
                    "loss": record.get("loss", ""),
                })

    csv_path = results_root / "task2_bpb_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("vocab_size", "split", "step", "tokens_seen", "bpb", "loss"),
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, vocab_size in zip(axes, vocab_sizes):
        config, metrics = loaded[vocab_size]
        train = metrics["train"]
        validation = metrics["validation"]
        if not train or not validation:
            raise ValueError(f"Both train and validation metrics are required for vocab {vocab_size}")

        train_x = [record["tokens_seen"] / 1_000_000 for record in train]
        train_y = [record["bpb"] for record in train]
        val_x = [record["tokens_seen"] / 1_000_000 for record in validation]
        val_y = [record["bpb"] for record in validation]

        axis.plot(train_x, train_y, color="#4C78A8", alpha=0.22, linewidth=0.8, label="Train BPB (raw)")
        axis.plot(train_x, moving_average(train_y), color="#4C78A8", linewidth=1.8, label="Train BPB (10-step mean)")
        axis.plot(val_x, val_y, color="#E45756", marker="o", markersize=3.5, linewidth=1.5, label="Validation BPB")
        axis.set_title(f"Vocabulary size {vocab_size:,}")
        axis.set_xlabel("Training tokens seen (millions)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    axes[0].set_ylabel("Bits per byte (BPB)")
    fig.suptitle("Task 2: Training and Validation BPB", fontsize=13)
    fig.tight_layout()

    png_path = results_root / "task2_bpb_curves.png"
    pdf_path = results_root / "task2_bpb_curves.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {csv_path}")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
