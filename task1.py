#!/usr/bin/env python3
"""
Task 1: Tokenization

Reproduces the two nanochat tokenizer runs and generates the report evidence
required for Assignment 1.

Expected project layout:
ALLM-A1/
├── nanochat-master/
├── task1.py
├── logs/
├── results/
└── data/                  # optional; controlled by NANOCHAT_BASE_DIR

Usage:
    export NANOCHAT_BASE_DIR=/path/to/nanochat-data

    python task1.py train
    python task1.py analyze --merge-example "YOUR OWN EXAMPLE"
    # or:
    python task1.py all --merge-example "YOUR OWN EXAMPLE"
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
NANOCHAT_ROOT = PROJECT_ROOT / "nanochat-master"
LOG_DIR = PROJECT_ROOT / "logs" / "task1"
RESULT_DIR = PROJECT_ROOT / "results" / "task1"

VOCAB_SIZES = (8192, 32768)
DEFAULT_NUM_TRAIN_SHARDS = 5

ENGLISH_SAMPLE = (
    "Large language models learn statistical patterns from text. "
    "A tokenizer converts this sentence into a sequence of discrete token IDs "
    "that the neural network can process efficiently."
)

FAILURE_CASES = {
    "numbers": "Order 007 costs $12.34; reference 2026-09-08 and ID 123456789.",
    "source_code": "def square(x):\n    return x * x\nprint(square(17))",
    "non_english": "中文分词并不依赖空格。 العربية جميلة. Ελληνικά. 😀",
}


def nanochat_base_dir() -> Path:
    """Mirror nanochat's portable base-dir convention without hardcoding a user path."""
    return Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat")).expanduser()


def ensure_layout() -> None:
    if not NANOCHAT_ROOT.exists():
        raise FileNotFoundError(
            f"Expected nanochat repo at {NANOCHAT_ROOT}. "
            "Clone it there or update NANOCHAT_ROOT in this script."
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_logged(command: list[str], log_path: Path) -> None:
    """Run an original nanochat command, showing output and saving the same output to a log."""
    print("\n$", " ".join(command))
    print(f"  cwd: {NANOCHAT_ROOT}")
    print(f"  log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=NANOCHAT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def train(num_train_shards: int) -> None:
    """
    Reproduce Task 1 training using nanochat's own dataset downloader and tok_train.py.

    tok_train.py always writes to $NANOCHAT_BASE_DIR/tokenizer, so after each run
    we copy that tokenizer to $NANOCHAT_BASE_DIR/task1_tokenizers/vocab_<size>.
    The final active tokenizer remains the 32,768-token version for downstream tasks.
    """
    ensure_layout()
    base_dir = nanochat_base_dir()
    archive_root = base_dir / "task1_tokenizers"
    archive_root.mkdir(parents=True, exist_ok=True)

    print(f"NANOCHAT_BASE_DIR = {base_dir}")

    run_logged(
        [sys.executable, "-m", "nanochat.dataset", "-n", str(num_train_shards)],
        LOG_DIR / "dataset_download.log",
    )

    data_dir = base_dir / "base_data_climbmix"
    train_shards = sorted(
        p for p in data_dir.glob("shard_*.parquet")
        if p.name != "shard_06542.parquet"
    )[:num_train_shards]
    total_bytes = sum(p.stat().st_size for p in train_shards)
    print(
        f"\nTraining shards present: {len(train_shards)} | "
        f"compressed size: {total_bytes / 1_000_000:.1f} MB"
    )

    for vocab_size in VOCAB_SIZES:
        run_logged(
            [
                sys.executable,
                "-m",
                "scripts.tok_train",
                "--vocab-size",
                str(vocab_size),
            ],
            LOG_DIR / f"train_vocab_{vocab_size}.log",
        )

        active_dir = base_dir / "tokenizer"
        archive_dir = archive_root / f"vocab_{vocab_size}"
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        shutil.copytree(active_dir, archive_dir)
        print(f"Archived tokenizer -> {archive_dir}")

    print(
        "\nTask 1 training complete.\n"
        f"Both tokenizers: {archive_root}\n"
        f"Active downstream tokenizer: {base_dir / 'tokenizer'} (32,768 vocab)"
    )


def load_tokenizers():
    """Load the archived tokenizers through nanochat's original tokenizer class."""
    sys.path.insert(0, str(NANOCHAT_ROOT))
    from nanochat.tokenizer import RustBPETokenizer

    root = nanochat_base_dir() / "task1_tokenizers"
    tokenizers = {}
    for vocab_size in VOCAB_SIZES:
        path = root / f"vocab_{vocab_size}"
        if not (path / "tokenizer.pkl").exists():
            raise FileNotFoundError(
                f"Missing {path / 'tokenizer.pkl'}. Run `python task1.py train` first."
            )
        tokenizers[vocab_size] = RustBPETokenizer.from_directory(str(path))
    return tokenizers


def token_pieces(tokenizer, text: str) -> list[str]:
    pieces = []
    for token_id in tokenizer.encode(text):
        raw = tokenizer.decode_single_token_bytes(token_id)
        pieces.append(raw.decode("utf-8", errors="backslashreplace"))
    return pieces


def compression_row(tokenizer, vocab_size: int, text: str) -> dict[str, object]:
    ids = tokenizer.encode(text)
    chars = len(text)
    byte_count = len(text.encode("utf-8"))
    return {
        "vocab_size": vocab_size,
        "characters": chars,
        "utf8_bytes": byte_count,
        "tokens": len(ids),
        "tokens_per_character": len(ids) / chars,
        "bytes_per_token": byte_count / len(ids),
    }


def bpe_trace_for_piece(enc, piece: bytes):
    """
    Reconstruct the learned byte-pair merge sequence for one pre-tokenized piece.

    Counts reported here are occurrence counts inside this concrete example.
    They are not corpus-wide training frequencies, which nanochat's saved
    tokenizer does not retain.
    """
    ranks = enc._mergeable_ranks
    parts = [bytes([b]) for b in piece]
    trace = []

    while len(parts) > 1:
        candidates = []
        for i in range(len(parts) - 1):
            pair = parts[i] + parts[i + 1]
            rank = ranks.get(pair)
            if rank is not None:
                candidates.append((rank, pair))
        if not candidates:
            break

        rank, chosen = min(candidates, key=lambda x: x[0])
        left_right = None
        for i in range(len(parts) - 1):
            if parts[i] + parts[i + 1] == chosen:
                left_right = (parts[i], parts[i + 1])
                break

        new_parts = []
        count = 0
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i] + parts[i + 1] == chosen:
                new_parts.append(chosen)
                count += 1
                i += 2
            else:
                new_parts.append(parts[i])
                i += 1

        trace.append(
            {
                "rank": rank,
                "left": left_right[0],
                "right": left_right[1],
                "merged": chosen,
                "count_in_example": count,
                "state": list(new_parts),
            }
        )
        parts = new_parts

    return parts, trace


def printable_bytes(b: bytes) -> str:
    return b.decode("utf-8", errors="backslashreplace").replace("\n", "\\n")


def write_merge_trace(tokenizer, text: str, path: Path) -> None:
    import regex

    enc = tokenizer.enc
    pieces = regex.findall(enc._pat_str, text)

    lines = [
        "BPE merge trace",
        f"Example: {text!r}",
        "Counts below are occurrences of the selected merge inside this example.",
        "",
    ]

    reconstructed_ids = []
    step_num = 0
    for piece_index, piece_text in enumerate(pieces, start=1):
        piece = piece_text.encode("utf-8")
        final_parts, trace = bpe_trace_for_piece(enc, piece)

        lines.append(f"Piece {piece_index}: {piece_text!r}")
        lines.append("  start: " + " | ".join(printable_bytes(bytes([b])) for b in piece))

        for item in trace:
            step_num += 1
            state = " | ".join(printable_bytes(x) for x in item["state"])
            lines.append(
                f"  merge {step_num:02d}: "
                f"{printable_bytes(item['left'])!r} + {printable_bytes(item['right'])!r} "
                f"-> {printable_bytes(item['merged'])!r} "
                f"(learned rank={item['rank']}, count={item['count_in_example']})"
            )
            lines.append(f"             state: {state}")

        for part in final_parts:
            reconstructed_ids.append(enc._mergeable_ranks[part])
        lines.append("")

    expected_ids = tokenizer.encode(text)
    lines.append(f"Final token IDs (reconstructed): {reconstructed_ids}")
    lines.append(f"Final token IDs (tokenizer):     {expected_ids}")
    lines.append(f"Exact match: {reconstructed_ids == expected_ids}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(merge_example: str) -> None:
    ensure_layout()
    tokenizers = load_tokenizers()

    metrics_path = RESULT_DIR / "tokenizer_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "vocab_size",
            "characters",
            "utf8_bytes",
            "tokens",
            "tokens_per_character",
            "bytes_per_token",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for vocab_size, tokenizer in tokenizers.items():
            writer.writerow(compression_row(tokenizer, vocab_size, ENGLISH_SAMPLE))

    examples_path = RESULT_DIR / "tokenizer_examples.txt"
    lines = ["TASK 1 TOKENIZER EVIDENCE", "", "Fixed English sample:", ENGLISH_SAMPLE, ""]

    for vocab_size, tokenizer in tokenizers.items():
        lines.extend(
            [
                "=" * 72,
                f"VOCAB SIZE {vocab_size}",
                "=" * 72,
                "",
                "English token pieces:",
                repr(token_pieces(tokenizer, ENGLISH_SAMPLE)),
                "",
            ]
        )
        for label, text in FAILURE_CASES.items():
            ids = tokenizer.encode(text)
            pieces = token_pieces(tokenizer, text)
            lines.extend(
                [
                    f"[{label}]",
                    f"text: {text}",
                    f"token_count: {len(ids)}",
                    f"token_ids: {ids}",
                    f"pieces: {pieces}",
                    "",
                ]
            )

    examples_path.write_text("\n".join(lines), encoding="utf-8")

    merge_path = RESULT_DIR / "merge_trace.txt"
    write_merge_trace(tokenizers[32768], merge_example, merge_path)

    print("\nGenerated report evidence:")
    print(f"  {metrics_path}")
    print(f"  {examples_path}")
    print(f"  {merge_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 1 - Task 1 tokenizer experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="download CLIMBMix sample and train both tokenizers")
    p_train.add_argument(
        "--num-train-shards",
        type=int,
        default=DEFAULT_NUM_TRAIN_SHARDS,
        help=f"number of CLIMBMix train shards (default: {DEFAULT_NUM_TRAIN_SHARDS})",
    )

    p_analyze = sub.add_parser("analyze", help="generate Task 1 measurements/examples")
    p_analyze.add_argument(
        "--merge-example",
        required=True,
        help="your own original example for the BPE merge trace",
    )

    p_all = sub.add_parser("all", help="run training then analysis")
    p_all.add_argument(
        "--num-train-shards",
        type=int,
        default=DEFAULT_NUM_TRAIN_SHARDS,
    )
    p_all.add_argument(
        "--merge-example",
        required=True,
        help="your own original example for the BPE merge trace",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args.num_train_shards)
    elif args.command == "analyze":
        analyze(args.merge_example)
    elif args.command == "all":
        train(args.num_train_shards)
        analyze(args.merge_example)


if __name__ == "__main__":
    main()
