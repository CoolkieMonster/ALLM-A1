#!/usr/bin/env python3
"""Task 2: pre-train depth-2 models and build the report-ready BPB evidence."""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

from utils import (
    NANOCHAT,
    ROOT,
    available_tokenizers,
    latest_checkpoint_step,
    replace_symlink,
    run_logged,
    warn,
)

DATA_DIR = ROOT / "data" / "base_data_climbmix"
TASK1_CKPT = ROOT / "results" / "task1" / "checkpoints"
CHECKPOINT_DIR = ROOT / "results" / "task2" / "checkpoints"
REPORT_DIR = ROOT / "results" / "task2" / "report"
LOG_DIR = ROOT / "logs" / "task2"

VOCAB_SIZES = (8_192, 32_768)
DEPTH = 2
MODEL_TAG = "d2"
DEVICE_BATCH_SIZE = 4
EVAL_EVERY = 50
EVAL_TOKENS = 1_048_576
TRAIN_BPB_EMA = 0.9
WANDB_RUN_PREFIX = "dummy"  # change to a name to enable W&B
PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
]


def prepare_run(vocab_size: int) -> Path:
    """Create one isolated nanochat base directory for a tokenizer/model pair."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Missing CLIMBMix data: {DATA_DIR}. Run task1_train.py first.")
    tokenizer = TASK1_CKPT / f"tokenizer_{vocab_size}"
    if not (tokenizer / "tokenizer.pkl").is_file():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer}")

    run_dir = CHECKPOINT_DIR / f"vocab_{vocab_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    replace_symlink(run_dir / "base_data_climbmix", DATA_DIR)
    replace_symlink(run_dir / "tokenizer", tokenizer)
    return run_dir


def train_models() -> list[int]:
    """Train every available Task-1 tokenizer; a missing/failed alternative is skipped."""
    print("\n[Task 2] Pre-training")
    sizes = available_tokenizers(TASK1_CKPT, VOCAB_SIZES)
    if not sizes:
        raise FileNotFoundError("No Task-1 tokenizer is available. Run task1_train.py first.")

    completed = []
    for vocab_size in sizes:
        print(f"\n[Task 2] Training depth-{DEPTH} model with vocab {vocab_size:,}")
        run_dir = prepare_run(vocab_size)
        model_dir = run_dir / "base_checkpoints" / MODEL_TAG
        if model_dir.exists():
            shutil.rmtree(model_dir)

        run_name = "dummy" if WANDB_RUN_PREFIX == "dummy" else f"{WANDB_RUN_PREFIX}-v{vocab_size}"
        cmd = [
            sys.executable, "-m", "scripts.base_train",
            "--depth", str(DEPTH),
            "--device-batch-size", str(DEVICE_BATCH_SIZE),
            "--model-tag", MODEL_TAG,
            "--eval-every", str(EVAL_EVERY),
            "--eval-tokens", str(EVAL_TOKENS),
            "--save-every", str(EVAL_EVERY),
            "--core-metric-every", "-1",
            "--sample-every", "-1",
            "--run", run_name,
        ]
        try:
            run_logged(cmd, LOG_DIR / f"pretrain_vocab_{vocab_size}.log", run_dir)
            completed.append(vocab_size)
        except Exception as exc:
            warn(f"pre-training failed for vocab {vocab_size:,}: {exc}")

    if not completed:
        raise RuntimeError("No Task-2 pre-training run completed successfully.")
    return completed


def parse_bpb_logs(vocab_sizes: list[int]) -> list[dict]:
    """Read online train BPB and held-out validation BPB from base_train logs."""
    print("\n[Task 2] Collecting BPB logs")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    with (REPORT_DIR / "bpb_metrics.jsonl").open("w", encoding="utf-8") as jsonl:
        for vocab_size in vocab_sizes:
            log_path = LOG_DIR / f"pretrain_vocab_{vocab_size}.log"
            if not log_path.is_file():
                warn(f"missing training log for vocab {vocab_size:,}; skipping BPB report")
                continue
            text = log_path.read_text(encoding="utf-8")
            batch_match = re.search(r"Total batch size\s+([\d,]+)", text)
            if not batch_match:
                warn(f"could not parse optimizer batch size for vocab {vocab_size:,}")
                continue
            total_batch = int(batch_match.group(1).replace(",", ""))

            config = {
                "record": "config",
                "vocab_size": vocab_size,
                "model_tag": MODEL_TAG,
                "total_batch_size": total_batch,
                "validation_eval_tokens": EVAL_TOKENS,
                "train_bpb_source": "actual optimizer batch",
                "validation_bpb_source": "fixed held-out validation prefix",
                "plot_train_ema_beta": TRAIN_BPB_EMA,
            }
            jsonl.write(json.dumps(config) + "\n")

            train_points = {
                int(m.group(1)): float(m.group(2))
                for m in re.finditer(
                    r"^step\s+(\d+)/(?:\d+).*?train bpb:\s*([\d.]+)", text, re.MULTILINE
                )
            }
            val_points = {
                int(m.group(1)): float(m.group(2))
                for m in re.finditer(
                    r"^Step\s+(\d+)\s+\|\s+Validation bpb:\s*([\d.]+)", text, re.MULTILINE
                )
            }

            if not train_points:
                warn(f"no online train BPB found for vocab {vocab_size:,}")
            if not val_points:
                warn(f"no validation BPB found for vocab {vocab_size:,}")

            for split, points in (("train", train_points), ("validation", val_points)):
                for step, value in sorted(points.items()):
                    row = {
                        "record": "metric",
                        "vocab_size": vocab_size,
                        "split": split,
                        "step": step,
                        "tokens_seen": step * total_batch,
                        "bpb": value,
                    }
                    rows.append(row)
                    jsonl.write(json.dumps(row) + "\n")

    if not rows:
        raise RuntimeError("No BPB metrics could be parsed from Task-2 logs.")

    fields = ["vocab_size", "split", "step", "tokens_seen", "bpb"]
    with (REPORT_DIR / "bpb_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    return rows


def plot_bpb(rows: list[dict], vocab_sizes: list[int]) -> None:
    """Plot smoothed train BPB and validation BPB at the same optimizer steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active = [v for v in vocab_sizes if any(r["vocab_size"] == v for r in rows)]
    fig, axes = plt.subplots(1, len(active), figsize=(5 * len(active), 3.8), sharey=True)
    if len(active) == 1:
        axes = [axes]

    for axis, vocab_size in zip(axes, active):
        subset = [r for r in rows if r["vocab_size"] == vocab_size]
        train = sorted((r for r in subset if r["split"] == "train"), key=lambda r: r["step"])
        val = {r["step"]: r["bpb"] for r in subset if r["split"] == "validation"}

        # Smooth online train BPB, then sample it only where validation was measured.
        smooth, ema = {}, None
        for row in train:
            ema = row["bpb"] if ema is None else TRAIN_BPB_EMA * ema + (1 - TRAIN_BPB_EMA) * row["bpb"]
            smooth[row["step"]] = ema
        steps = sorted(set(smooth) & set(val))

        axis.plot(steps, [smooth[s] for s in steps], marker="o", linewidth=1.3,
                  markersize=3.0, label="Train (EMA)")
        axis.plot(steps, [val[s] for s in steps], marker="s", linewidth=1.3,
                  markersize=3.0, label="Validation")
        axis.set_title(f"Vocabulary {vocab_size:,}")
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Bits per byte (BPB)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "bpb_curves.pdf", bbox_inches="tight")
    fig.savefig(REPORT_DIR / "bpb_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def scaling_summary(vocab_sizes: list[int]) -> None:
    """Compare nanochat's 12*N_s horizon with the approximate Chinchilla 20*N_total rule."""
    print("\n[Task 2] Scaling summary")
    rows, blocks = [], ["Task 2 scaling-law calculation", ""]

    for vocab_size in vocab_sizes:
        log_path = LOG_DIR / f"pretrain_vocab_{vocab_size}.log"
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8")

        def number(pattern: str) -> int:
            match = re.search(pattern, text, re.MULTILINE)
            if not match:
                raise ValueError(f"Could not parse {pattern!r} from {log_path}")
            return int(match.group(1).replace(",", ""))

        total = number(r"^total\s*:\s*([\d,]+)")
        matrices = number(r"^transformer_matrices\s*:\s*([\d,]+)")
        lm_head = number(r"^lm_head\s*:\s*([\d,]+)")
        actual_tokens = number(r"Total number of training tokens:\s*([\d,]+)")
        total_batch = number(r"Total batch size\s+([\d,]+)")
        scaling = matrices + lm_head
        nanochat_target = 12 * scaling
        chinchilla_target = 20 * total

        row = {
            "vocab_size": vocab_size,
            "total_parameters": total,
            "scaling_parameters": scaling,
            "nanochat_12Ns": nanochat_target,
            "optimizer_batch_tokens": total_batch,
            "actual_training_tokens": actual_tokens,
            "chinchilla_20Ntotal": chinchilla_target,
            "actual_over_chinchilla": actual_tokens / chinchilla_target,
        }
        rows.append(row)
        blocks += [
            f"Vocabulary {vocab_size:,}",
            f"  N_s = transformer_matrices + lm_head = {matrices:,} + {lm_head:,} = {scaling:,}",
            f"  Nanochat target 12*N_s = {nanochat_target:,} tokens",
            f"  Optimizer batch B      = {total_batch:,} tokens",
            f"  Actual horizon D       = {actual_tokens:,} tokens",
            f"  Total parameters N     = {total:,}",
            f"  Chinchilla ~20*N       = {chinchilla_target:,} tokens",
            f"  D / Chinchilla target  = {actual_tokens / chinchilla_target:.3f}",
            "",
        ]

    if not rows:
        warn("no scaling summary could be produced")
        return
    (REPORT_DIR / "scaling_summary.txt").write_text("\n".join(blocks), encoding="utf-8")
    with (REPORT_DIR / "scaling_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def completions(vocab_sizes: list[int]) -> None:
    """Sample five raw next-token completions from each final pre-trained model."""
    print("\n[Task 2] Raw completions")
    blocks = ["Task 2 raw next-token completions", "temperature=0, max_new_tokens=16", ""]
    csv_rows = []

    for vocab_size in vocab_sizes:
        run_dir = prepare_run(vocab_size)
        ckpt_dir = run_dir / "base_checkpoints" / MODEL_TAG
        final_step = latest_checkpoint_step(ckpt_dir)
        if final_step is None:
            warn(f"no checkpoint for vocab {vocab_size:,}; skipping completions")
            continue
        raw = run_logged(
            [
                sys.executable, "-m", "scripts.base_eval", "--eval", "sample",
                "--model-tag", MODEL_TAG,
                "--step", str(final_step),
                "--device-batch-size", str(DEVICE_BATCH_SIZE),
            ],
            LOG_DIR / f"completions_vocab_{vocab_size}.log",
            run_dir,
            capture=True,
        )
        blocks += ["=" * 72, f"VOCABULARY {vocab_size:,} | CHECKPOINT STEP {final_step}", "=" * 72]
        for i, prompt in enumerate(PROMPTS, start=1):
            match = re.search(
                re.escape(prompt) + r"(.*?)(?=\n-{20,}|\n\nUnconditioned samples:|\Z)",
                raw,
                re.DOTALL,
            )
            prediction = match.group(1).strip() if match else "[see raw log]"
            blocks += [f"[{i}] Prompt: {prompt}", f"    Prediction: {prediction}", ""]
            csv_rows.append({
                "vocab_size": vocab_size,
                "step": final_step,
                "prompt": prompt,
                "prediction": prediction,
            })

    if not csv_rows:
        warn("no completions were generated")
        return
    (REPORT_DIR / "completions.txt").write_text("\n".join(blocks), encoding="utf-8")
    with (REPORT_DIR / "completions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    completed = train_models()
    rows = parse_bpb_logs(completed)
    plot_bpb(rows, completed)
    scaling_summary(completed)
    completions(completed)
    print(f"\n[Task 2] Done: {REPORT_DIR}")


if __name__ == "__main__":
    main()
