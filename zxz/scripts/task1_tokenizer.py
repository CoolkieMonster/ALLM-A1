"""Reproduce Task 1: train 8K and 32K BPE tokenizers on the same CLIMBMix sample."""

import time
from pathlib import Path

import torch

from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer


SAMPLE_BYTES = 500_000_000  # 500 MB of UTF-8 text (decimal MB)
DOC_CAP = 10_000            # match nanochat's default per-document cap
VOCAB_SIZES = (8_192, 32_768)


def text_iterator():
    """Yield the first SAMPLE_BYTES of the deterministic CLIMBMix train split."""
    total_bytes = 0

    def report_total():
        print(
            f"total_bytes: {total_bytes:,} bytes "
            f"({total_bytes / 1_000_000:.2f} MB)"
        )

    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            text = doc[:DOC_CAP]
            raw = text.encode("utf-8")
            remaining = SAMPLE_BYTES - total_bytes
            if remaining <= 0:
                report_total()
                return
            if len(raw) > remaining:
                # Avoid cutting through a UTF-8 code point at the byte boundary.
                text = raw[:remaining].decode("utf-8", errors="ignore")
                if text:
                    total_bytes += len(text.encode("utf-8"))
                    yield text
                report_total()
                return
            yield text
            total_bytes += len(raw)

    report_total()


def save_token_bytes(tokenizer, output_dir: Path):
    """Save the byte length for every token ID for later BPB evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = tokenizer.get_vocab_size()
    special_ids = {
        tokenizer.encode_special(token)
        for token in tokenizer.get_special_tokens()
    }
    token_bytes = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            token_bytes.append(0)
        else:
            token_bytes.append(
                len(tokenizer.decode_single_token_bytes(token_id))
            )
    torch.save(
        torch.tensor(token_bytes, dtype=torch.int32),
        output_dir / "token_bytes.pt",
    )


def train_one(vocab_size: int, output_root: Path):
    print(f"\nTraining vocab_size={vocab_size:,}")
    started = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(
        text_iterator(), vocab_size
    )
    output_dir = output_root / f"tokenizer_{vocab_size}"
    tokenizer.save(str(output_dir))
    save_token_bytes(tokenizer, output_dir)

    test_text = """Hello world! This is a test.
    Numbers: 123, 4567, 89
    Contractions: I'm, you're, it's
    Special chars: @#$%^&*()
    Unicode: 你好世界 🌍"""
    assert tokenizer.decode(tokenizer.encode(test_text)) == test_text
    print(f"Saved to {output_dir}")
    print(f"Training time: {time.time() - started:.2f}s")


if __name__ == "__main__":
    base_dir = Path(get_base_dir())
    output_root = base_dir / "task1_tokenizers"
    print(f"NANOCHAT_BASE_DIR: {base_dir}")
    print(f"Output directory: {output_root}")
    print(f"DataSet: CLIMBMix,Sample size: {SAMPLE_BYTES:,} UTF-8 bytes")
    print(f"Document cap: {DOC_CAP:,} characters")
    for vocab_size in VOCAB_SIZES:
        train_one(vocab_size, output_root)
