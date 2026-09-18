#!/usr/bin/env bash
set -euo pipefail

# Always run from the nanochat repository root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT_ROOT="$PWD"
RESULTS_ROOT="$PROJECT_ROOT/results"
TASK1_TOKENIZERS_DIR="$RESULTS_ROOT/task1_tokenizers"
SHARED_DATA_DIR="$RESULTS_ROOT/base_data_climbmix"
TASK2_OUTPUT_DIR="$RESULTS_ROOT/task2_models"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_TOKENS="${EVAL_TOKENS:-1048576}"
MODEL_TAG="${MODEL_TAG:-d2_bpb}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ ! -d "$SHARED_DATA_DIR" ]]; then
    echo "Error: training data not found: $SHARED_DATA_DIR" >&2
    echo "Run bash runs/task1_tokenizer.sh first." >&2
    exit 1
fi

mkdir -p "$TASK2_OUTPUT_DIR"

for vocab_size in 8192 32768; do
    tokenizer_dir="$TASK1_TOKENIZERS_DIR/tokenizer_${vocab_size}"
    run_dir="$TASK2_OUTPUT_DIR/vocab_${vocab_size}"

    if [[ ! -f "$tokenizer_dir/tokenizer.pkl" || ! -f "$tokenizer_dir/token_bytes.pt" ]]; then
        echo "Error: tokenizer artifacts not found in: $tokenizer_dir" >&2
        echo "Run bash runs/task1_tokenizer.sh first." >&2
        exit 1
    fi

    mkdir -p "$run_dir"

    # base_train reads both paths relative to NANOCHAT_BASE_DIR. Keep the large
    # dataset shared, but give each tokenizer/model run its own output directory.
    for path_name in base_data base_data_climbmix tokenizer; do
        path="$run_dir/$path_name"
        if [[ -e "$path" && ! -L "$path" ]]; then
            echo "Error: $path already exists and is not a symbolic link." >&2
            exit 1
        fi
    done
    ln -sfn "$SHARED_DATA_DIR" "$run_dir/base_data"
    ln -sfn "$SHARED_DATA_DIR" "$run_dir/base_data_climbmix"
    ln -sfn "$tokenizer_dir" "$run_dir/tokenizer"

    export NANOCHAT_BASE_DIR="$run_dir"

    echo
    echo "============================================================"
    echo "Training depth-2 model with vocabulary size $vocab_size"
    echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
    echo "Checkpoint directory: $run_dir/base_checkpoints/$MODEL_TAG"
    echo "Device batch size: $DEVICE_BATCH_SIZE"
    echo "Validation interval: every $EVAL_EVERY steps"
    echo "Validation sample: $EVAL_TOKENS tokens"
    echo "Log file: $run_dir/base_train_bpb.log"
    echo "Metrics file: $run_dir/base_train_bpb_metrics.jsonl"
    echo "============================================================"

    # Depth remains the only model/training-scale dial. The other flags below
    # only control measurement frequency, structured output, and the checkpoint
    # name; they do not override the automatically derived architecture, batch
    # size, learning rates, or training horizon.
    python -m scripts.base_train \
        --depth=2 \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --model-tag="$MODEL_TAG" \
        --eval-every="$EVAL_EVERY" \
        --eval-tokens="$EVAL_TOKENS" \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --log-train-bpb \
        --metrics-jsonl="$run_dir/base_train_bpb_metrics.jsonl" \
        2>&1 | tee "$run_dir/base_train_bpb.log"
done

python -m scripts.plot_task2_bpb --results-root "$TASK2_OUTPUT_DIR"

echo
echo "Task 2 complete. Results are in: $TASK2_OUTPUT_DIR"
echo "BPB plot: $TASK2_OUTPUT_DIR/task2_bpb_curves.png"
echo "BPB table: $TASK2_OUTPUT_DIR/task2_bpb_metrics.csv"
