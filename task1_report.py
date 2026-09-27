#!/usr/bin/env python3
"""Task 1: build report-ready tokenizer measurements from available checkpoints."""

from __future__ import annotations

import ast
import csv
import os
import sys
from collections import Counter
from pathlib import Path

from utils import NANOCHAT, ROOT, available_tokenizers, warn

CHECKPOINT_DIR = ROOT / "results" / "task1" / "checkpoints"
REPORT_DIR = ROOT / "results" / "task1" / "report"
VOCAB_SIZES = (8_192, 32_768)
MERGE_EXAMPLE = "tokenizer tokenizes tokens"
NUMBERS_TEXT = "Order 007 costs $12.34; reference 2026-09-08 and ID 123456789."
TOK_EVAL_NAMES = {
    "news_text": "news_english",
    "science_text": "science_english",
    "code_text": "source_code",
    "korean_text": "non_english_korean",
    "math_text": "math_latex",
}


def load_inputs():
    """Load every available Task-1 tokenizer and nanochat's fixed test texts."""
    os.environ["NANOCHAT_BASE_DIR"] = str(ROOT / "data")
    sys.path.insert(0, str(NANOCHAT))
    from nanochat.tokenizer import RustBPETokenizer

    sizes = available_tokenizers(CHECKPOINT_DIR, VOCAB_SIZES)
    if not sizes:
        raise FileNotFoundError("No Task-1 tokenizer found. Run task1_train.py first.")
    tokenizers = {
        size: RustBPETokenizer.from_directory(str(CHECKPOINT_DIR / f"tokenizer_{size}"))
        for size in sizes
    }

    tree = ast.parse((NANOCHAT / "scripts" / "tok_eval.py").read_text(encoding="utf-8"))
    texts = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in TOK_EVAL_NAMES:
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "strip":
            value = value.func.value
        texts[TOK_EVAL_NAMES[name]] = ast.literal_eval(value).strip()
    if set(texts) != set(TOK_EVAL_NAMES.values()):
        raise RuntimeError("Could not recover all fixed texts from scripts/tok_eval.py")
    return tokenizers, texts


def compression_analysis(tokenizers, texts) -> None:
    """Measure sequence length and compression on the same visible text samples."""
    with (REPORT_DIR / "compression_texts.txt").open("w", encoding="utf-8") as f:
        for name, text in texts.items():
            f.write(f"===== {name} =====\n{text}\n\n")

    fields = ["text", "vocab_size", "characters", "utf8_bytes", "tokens",
              "tokens_per_character", "bytes_per_token"]
    with (REPORT_DIR / "compression_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, text in texts.items():
            for vocab_size, tokenizer in tokenizers.items():
                ids = tokenizer.encode(text)
                writer.writerow({
                    "text": name,
                    "vocab_size": vocab_size,
                    "characters": len(text),
                    "utf8_bytes": len(text.encode("utf-8")),
                    "tokens": len(ids),
                    "tokens_per_character": len(ids) / len(text),
                    "bytes_per_token": len(text.encode("utf-8")) / len(ids),
                })


def failure_analysis(tokenizers, texts) -> None:
    """Record token fragmentation for numbers, code, non-English text and LaTeX."""
    cases = {
        "numbers": NUMBERS_TEXT,
        "source_code": texts["source_code"],
        "non_english_korean": texts["non_english_korean"],
        "math_latex": texts["math_latex"],
    }
    rows, blocks = [], []
    for label, text in cases.items():
        blocks.append(f"===== {label} =====\nTEXT:\n{text}\n")
        for vocab_size, tokenizer in tokenizers.items():
            ids = tokenizer.encode(text)
            pieces = [
                tokenizer.decode_single_token_bytes(i).decode("utf-8", errors="backslashreplace")
                for i in ids
            ]
            rows.append({
                "case": label,
                "vocab_size": vocab_size,
                "characters": len(text),
                "utf8_bytes": len(text.encode("utf-8")),
                "tokens": len(ids),
                "tokens_per_character": len(ids) / len(text),
                "bytes_per_token": len(text.encode("utf-8")) / len(ids),
            })
            blocks.append(
                f"VOCAB {vocab_size:,}\n"
                f"token_count: {len(ids)}\n"
                f"token_ids: {ids}\n"
                f"pieces: {pieces}\n"
            )
        blocks.append("")

    (REPORT_DIR / "failure_cases.txt").write_text("\n".join(blocks), encoding="utf-8")
    with (REPORT_DIR / "failure_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def merge_example(tokenizers) -> None:
    """Produce a compact BPE teaching trace with all pair candidates and counts."""
    # Keep words separate so the example does not merge across spaces.
    words = [[ch for ch in word] for word in MERGE_EXAMPLE.split()]
    lines = [
        "BPE merge example",
        f"Text: {MERGE_EXAMPLE}",
        "At each iteration, count adjacent pairs and merge one most frequent pair.",
        "Ties are resolved by first appearance in the current text.",
        "",
    ]

    iteration = 0
    while True:
        pairs = []
        first_seen = {}
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in first_seen:
                    first_seen[pair] = len(first_seen)
                pairs.append(pair)
        counts = Counter(pairs)
        if not counts:
            break
        best_count = max(counts.values())
        if best_count < 2:
            break
        chosen = min((p for p, c in counts.items() if c == best_count), key=first_seen.get)
        iteration += 1

        lines.append(f"Iteration {iteration}")
        for pair in sorted(counts, key=first_seen.get):
            mark = "  <-- selected" if pair == chosen else ""
            lines.append(f"  ({pair[0]!r}, {pair[1]!r}) : {counts[pair]}{mark}")

        merged = chosen[0] + chosen[1]
        new_words = []
        for word in words:
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == chosen:
                    out.append(merged)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            new_words.append(out)
        words = new_words
        lines.append("  state: " + "   ".join(" | ".join(word) for word in words))
        lines.append("")

    lines.append("Stop: no adjacent pair occurs more than once in this short example.")
    lines.append("")
    lines.append("Tokenization by the trained checkpoints:")
    for vocab_size, tokenizer in tokenizers.items():
        ids = tokenizer.encode(MERGE_EXAMPLE)
        pieces = [
            tokenizer.decode_single_token_bytes(i).decode("utf-8", errors="backslashreplace")
            for i in ids
        ]
        lines.append(f"  {vocab_size:,}: {len(ids)} tokens -> {pieces}")

    (REPORT_DIR / "merge_example.txt").write_text(MERGE_EXAMPLE + "\n", encoding="utf-8")
    (REPORT_DIR / "merge_trace.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("[Task 1 report] Loading tokenizer outputs")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizers, texts = load_inputs()
    if len(tokenizers) == 1:
        warn("only one tokenizer is available; cross-vocabulary comparisons will contain one run")

    print("[Task 1 report] Compression analysis")
    compression_analysis(tokenizers, texts)
    print("[Task 1 report] Failure cases")
    failure_analysis(tokenizers, texts)
    print("[Task 1 report] BPE merge example")
    merge_example(tokenizers)
    print(f"[Task 1 report] Done: {REPORT_DIR}")


if __name__ == "__main__":
    main()
