#!/usr/bin/env python3
"""Task 3: inspect data, run mid-training then SFT, and benchmark all stages."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

from utils import NANOCHAT, ROOT, replace_symlink, run_logged, select_task2_run

TASK2_CKPT = ROOT / "results" / "task2" / "checkpoints"
CHECKPOINT_DIR = ROOT / "results" / "task3" / "checkpoints"
REPORT_DIR = ROOT / "results" / "task3" / "report"
LOG_DIR = ROOT / "logs" / "task3"

BASE_TAG = "d2"
MID_TAG = "midtrain-d2"
SFT_TAG = "sft-d2"
DEVICE_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
DATASET_EXAMPLES = 3
BENCHMARKS = "ARC-Easy|ARC-Challenge|GSM8K"


def prepare_runtime() -> int:
    """Use the preferred Task-2 model, falling back to another complete run."""
    vocab_size, run_dir = select_task2_run(TASK2_CKPT, preferred_vocab=32_768, model_tag=BASE_TAG)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    replace_symlink(CHECKPOINT_DIR / "base_checkpoints", run_dir / "base_checkpoints")
    replace_symlink(CHECKPOINT_DIR / "tokenizer", run_dir / "tokenizer")
    os.environ["NANOCHAT_BASE_DIR"] = str(CHECKPOINT_DIR)
    print(f"[Task 3] Base model: vocab {vocab_size:,} from {run_dir.name}")
    return vocab_size


def inspect_data() -> None:
    """Record dataset sizes and a few visible examples for both stages."""
    print("\n[Task 3] Inspecting MMLU, GSM8K and SmolTalk")
    sys.path.insert(0, str(NANOCHAT))
    from tasks.mmlu import MMLU
    from tasks.gsm8k import GSM8K
    from tasks.smoltalk import SmolTalk

    datasets = {
        "MMLU auxiliary_train": (MMLU(subset="all", split="auxiliary_train"), 3, "midtrain"),
        "GSM8K train": (GSM8K(subset="main", split="train"), 4, "midtrain"),
        "SmolTalk train": (SmolTalk(split="train"), 1, "sft"),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "dataset_sizes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "stage", "raw_examples", "repetitions", "stage_contribution"])
        for name, (dataset, repetitions, stage) in datasets.items():
            writer.writerow([name, stage, len(dataset), repetitions, len(dataset) * repetitions])

    with (REPORT_DIR / "dataset_examples.txt").open("w", encoding="utf-8") as f:
        for name, (dataset, repetitions, stage) in datasets.items():
            f.write(f"===== {name} =====\n")
            f.write(f"stage={stage}, raw_examples={len(dataset):,}, repetitions={repetitions}\n")
            for i in range(min(DATASET_EXAMPLES, len(dataset))):
                f.write(f"\nExample {i + 1}:\n{json.dumps(dataset[i], ensure_ascii=False, indent=2)}\n")
            f.write("\n")


def evaluate(stage: str) -> None:
    """Run the same ARC-Easy, ARC-Challenge and GSM8K evaluation for one stage."""
    source, tag = {
        "base": ("base", BASE_TAG),
        "midtrain": ("sft", MID_TAG),
        "sft": ("sft", SFT_TAG),
    }[stage]
    print(f"\n[Task 3] Evaluating {stage}")
    run_logged(
        [
            sys.executable, "-m", "scripts.chat_eval",
            "--source", source,
            "--model-tag", tag,
            "--task-name", BENCHMARKS,
            "--batch-size", str(EVAL_BATCH_SIZE),
            "--temperature", "0",
        ],
        LOG_DIR / f"{stage}_eval.log",
        CHECKPOINT_DIR,
    )


def train_stage(stage: str) -> None:
    """Run one staged adaptation pass with the modified nanochat chat_sft.py."""
    input_tag, output_tag = {
        "midtrain": (BASE_TAG, MID_TAG),
        "sft": (MID_TAG, SFT_TAG),
    }[stage]
    output_dir = CHECKPOINT_DIR / "chatsft_checkpoints" / output_tag
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"\n[Task 3] Training stage: {stage}")
    run_logged(
        [
            sys.executable, "-m", "scripts.chat_sft",
            "--stage", stage,
            "--input-tag", input_tag,
            "--output-tag", output_tag,
            "--device-batch-size", str(DEVICE_BATCH_SIZE),
            "--chatcore-every", "-1",
            "--run", "dummy",
        ],
        LOG_DIR / f"{stage}.log",
        CHECKPOINT_DIR,
    )


def summarize_benchmarks() -> None:
    """Combine the three benchmark logs into one report-ready table."""
    print("\n[Task 3] Summarizing benchmark scores")
    tasks = ["ARC-Easy", "ARC-Challenge", "GSM8K"]
    rows = []
    for stage in ("base", "midtrain", "sft"):
        path = LOG_DIR / f"{stage}_eval.log"
        text = path.read_text(encoding="utf-8")
        scores = {
            m.group(1): float(m.group(2))
            for m in re.finditer(
                r"^(ARC-Easy|ARC-Challenge|GSM8K) accuracy:\s*([\d.]+)%",
                text,
                re.MULTILINE,
            )
        }
        if set(scores) != set(tasks):
            raise ValueError(f"Could not parse all benchmark scores from {path}")
        rows.append([stage] + [scores[t] for t in tasks])

    with (REPORT_DIR / "benchmark_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage"] + tasks)
        writer.writerows(rows)

    lines = [
        "Task 3 benchmark accuracy (%)",
        "evaluation temperature=0.0 (greedy for generative tasks)",
        "",
        f"{'stage':<12}{'ARC-Easy':>12}{'ARC-Challenge':>16}{'GSM8K':>12}",
    ]
    for row in rows:
        lines.append(f"{row[0]:<12}{row[1]:>12.2f}{row[2]:>16.2f}{row[3]:>12.2f}")
    (REPORT_DIR / "benchmark_scores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prepare_runtime()

    # Stage 0: inspect the data and evaluate the pretrained base model.
    inspect_data()
    evaluate("base")

    # Stage 1: reasoning/knowledge mid-training, then benchmark it.
    train_stage("midtrain")
    evaluate("midtrain")

    # Stage 2: SmolTalk instruction SFT from the Stage-1 checkpoint, then benchmark it.
    train_stage("sft")
    evaluate("sft")

    summarize_benchmarks()
    print(f"\n[Task 3] Done: {REPORT_DIR}")


if __name__ == "__main__":
    main()
