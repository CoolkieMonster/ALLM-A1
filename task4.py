#!/usr/bin/env python3
"""Task 4: sample five fixed prompts at temperatures 0.1, 0.7 and 1.5."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from utils import NANOCHAT, ROOT, latest_checkpoint_step, select_task2_run, warn

TASK2_CKPT = ROOT / "results" / "task2" / "checkpoints"
TASK3_CKPT = ROOT / "results" / "task3" / "checkpoints"
REPORT_DIR = ROOT / "results" / "task4" / "report"
LOG_DIR = ROOT / "logs" / "task4"

TEMPERATURES = (0.1, 0.7, 1.5)
TOP_K = 50
SEED = 42
PROMPTS = [
    ("P1", "What is the capital of France?\nAnswer in one sentence.", 64),
    ("P2", "Why does rain fall from clouds?\nExplain in two short sentences.", 96),
    ("P3", "A box contains 12 apples. If 5 are removed, how many remain? Explain briefly.", 96),
    ("P4", "Write a Python function that returns the larger of two numbers.", 128),
    ("P5", "Write a short story about a robot finding a flower.", 192),
]


def choose_model() -> tuple[Path, str, str]:
    """Prefer final SFT, then mid-training, then a Task-2 base model."""
    tokenizer = TASK3_CKPT / "tokenizer" / "tokenizer.pkl"
    for tag in ("sft-d2", "midtrain-d2"):
        model_dir = TASK3_CKPT / "chatsft_checkpoints" / tag
        if tokenizer.is_file() and latest_checkpoint_step(model_dir) is not None:
            if tag != "sft-d2":
                warn("final SFT checkpoint is unavailable; using the mid-training checkpoint")
            return TASK3_CKPT, "sft", tag

    base_dir = TASK3_CKPT / "base_checkpoints" / "d2"
    if tokenizer.is_file() and latest_checkpoint_step(base_dir) is not None:
        warn("Task-3 adapted checkpoints are unavailable; using the linked pretrained base model")
        return TASK3_CKPT, "base", "d2"

    vocab_size, run_dir = select_task2_run(TASK2_CKPT, preferred_vocab=32_768, model_tag="d2")
    warn(f"Task-3 runtime is unavailable; using Task-2 base model with vocab {vocab_size:,}")
    return run_dir, "base", "d2"


def main() -> None:
    print("[Task 4] Loading model")
    base_dir, source, model_tag = choose_model()
    os.environ["NANOCHAT_BASE_DIR"] = str(base_dir)
    sys.path.insert(0, str(NANOCHAT))

    from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag)
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "Task 4 temperature sampling",
        f"source={source}",
        f"model_tag={model_tag}",
        f"checkpoint_step={meta['step']}",
        f"top_k={TOP_K}",
        f"seed={SEED}",
        "",
    ]

    for label, prompt, max_tokens in PROMPTS:
        print(f"\n[Task 4] {label}")
        prefix = [bos, user_start]
        prefix.extend(tokenizer.encode(prompt))
        prefix.extend([user_end, assistant_start])

        lines = [
            label,
            "Prompt:",
            prompt,
            "",
            f"Model: {source}/{model_tag}, checkpoint step {meta['step']}",
            f"top_k={TOP_K}, seed={SEED}, max_new_tokens={max_tokens}",
            "",
        ]
        for temperature in TEMPERATURES:
            result, _ = engine.generate_batch(
                prefix,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=TOP_K,
                seed=SEED,
            )
            response_ids = result[0][len(prefix):]
            response = tokenizer.decode(response_ids).strip()
            lines += [f"Temperature {temperature}", response, ""]
            log_lines += [f"{label} | temperature={temperature}", response, ""]
            print(f"  T={temperature}: {response[:120].replace(chr(10), ' ')}")

        (REPORT_DIR / f"{label}.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    (LOG_DIR / "generation.log").write_text("\n".join(log_lines), encoding="utf-8")
    compute_cleanup()
    print(f"\n[Task 4] Done: {REPORT_DIR}")


if __name__ == "__main__":
    main()
