#!/usr/bin/env bash
set -euo pipefail

# Always run from the nanochat repository root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Keep CLIMBMix data and Task 1 artifacts under the requested results directory.
export NANOCHAT_BASE_DIR="$PWD/results"
mkdir -p "$NANOCHAT_BASE_DIR"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"

echo "Downloading/checking CLIMBMix shards..."
# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8

echo "Running Task 1 tokenizer experiments..."
python -m scripts.task1_tokenizer

echo "Evaluating compression ratios on the fixed English news_text sample..."
TASK1_OUTPUT_DIR="$NANOCHAT_BASE_DIR/task1_tokenizers"
TASK1_TOKENIZER_LINK="$NANOCHAT_BASE_DIR/tokenizer"
TASK1_RATIO_LOG="$TASK1_OUTPUT_DIR/tok_eval_results.txt"
mkdir -p "$TASK1_OUTPUT_DIR"
: > "$TASK1_RATIO_LOG"

# scripts.tok_eval loads the tokenizer from $NANOCHAT_BASE_DIR/tokenizer.
# Point that path at each Task 1 tokenizer in turn, then remove the temporary link.
cleanup_tokenizer_link() {
    if [ -L "$TASK1_TOKENIZER_LINK" ]; then
        rm "$TASK1_TOKENIZER_LINK"
    fi
}
trap cleanup_tokenizer_link EXIT

for vocab_size in 8192 32768; do
    ln -sfn "task1_tokenizers/tokenizer_${vocab_size}" "$TASK1_TOKENIZER_LINK"
    printf '\n===== tokenizer_%s =====\n' "$vocab_size" | tee -a "$TASK1_RATIO_LOG"
    python -m scripts.tok_eval \
        | tee "$TASK1_OUTPUT_DIR/tok_eval_${vocab_size}.txt" \
        | tee -a "$TASK1_RATIO_LOG"
done

echo "Task 1 tokenizers are in: $TASK1_OUTPUT_DIR"
echo "Compression results are in: $TASK1_RATIO_LOG"
