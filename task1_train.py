#!/usr/bin/env python3
"""Task 1: train the 8K and 32K BPE tokenizers on the same 500 MB sample."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import torch

from utils import NANOCHAT, ROOT, run_logged, warn

DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "results" / "task1" / "checkpoints"
LOG_DIR = ROOT / "logs" / "task1"

SAMPLE_BYTES = 500_000_000
DOWNLOAD_SHARDS = 8
DOCUMENT_CAP = 10_000
VOCAB_SIZES = (8_192, 32_768)


def download_data() -> None:
    """Make the assignment CLIMBMix shards available under data/."""
    print("\n[Task 1] Checking CLIMBMix data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_logged(
        [sys.executable, "-m", "nanochat.dataset", "-n", str(DOWNLOAD_SHARDS)],
        LOG_DIR / "dataset_download.log",
        DATA_DIR,
    )


def train_tokenizers() -> list[int]:
    """Train both vocabulary sizes; one failed run does not discard the other."""
    os.environ["NANOCHAT_BASE_DIR"] = str(DATA_DIR)
    sys.path.insert(0, str(NANOCHAT))
    from nanochat.dataset import parquets_iter_batched
    from nanochat.tokenizer import RustBPETokenizer

    def text_iterator():
        total = 0
        for batch in parquets_iter_batched(split="train"):
            for document in batch:
                text = document[:DOCUMENT_CAP]
                raw = text.encode("utf-8")
                remaining = SAMPLE_BYTES - total
                if remaining <= 0:
                    return
                if len(raw) > remaining:
                    text = raw[:remaining].decode("utf-8", errors="ignore")
                    raw = text.encode("utf-8")
                if raw:
                    total += len(raw)
                    yield text
                if total >= SAMPLE_BYTES:
                    return
        raise RuntimeError(f"Only {total:,} bytes available; expected {SAMPLE_BYTES:,}.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    completed = []
    for vocab_size in VOCAB_SIZES:
        output = CHECKPOINT_DIR / f"tokenizer_{vocab_size}"
        if output.exists():
            shutil.rmtree(output)
        print(f"\n[Task 1] Training tokenizer {vocab_size:,}")
        started = time.time()
        try:
            tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), vocab_size)
            tokenizer.save(str(output))
            special_ids = {tokenizer.encode_special(t) for t in tokenizer.get_special_tokens()}
            token_bytes = [
                0 if i in special_ids else len(tokenizer.decode_single_token_bytes(i))
                for i in range(tokenizer.get_vocab_size())
            ]
            torch.save(torch.tensor(token_bytes, dtype=torch.int32), output / "token_bytes.pt")
            probe = "Tokenizer round-trip: 123, code(), 한국어, 中文, 😀"
            assert tokenizer.decode(tokenizer.encode(probe)) == probe
            completed.append(vocab_size)
            print(f"Saved {output} ({time.time() - started:.1f}s)")
        except Exception as exc:
            warn(f"tokenizer {vocab_size:,} failed: {exc}")
            if output.exists():
                shutil.rmtree(output)

    if not completed:
        raise RuntimeError("No tokenizer was trained successfully.")
    return completed


def write_manifest(completed: list[int]) -> None:
    manifest = CHECKPOINT_DIR / "training_manifest.txt"
    manifest.write_text(
        "Task 1 tokenizer training\n"
        "dataset=CLIMBMix train split\n"
        f"sample_utf8_bytes={SAMPLE_BYTES}\n"
        f"document_cap_characters={DOCUMENT_CAP}\n"
        f"downloaded_train_shards={DOWNLOAD_SHARDS}\n"
        f"requested_vocab_sizes={','.join(map(str, VOCAB_SIZES))}\n"
        f"completed_vocab_sizes={','.join(map(str, completed))}\n",
        encoding="utf-8",
    )


def main() -> None:
    if not NANOCHAT.is_dir():
        raise FileNotFoundError(f"nanochat repository not found: {NANOCHAT}")
    download_data()
    completed = train_tokenizers()
    write_manifest(completed)
    print(f"\n[Task 1] Done: {', '.join(f'{v:,}' for v in completed)}")


if __name__ == "__main__":
    main()
