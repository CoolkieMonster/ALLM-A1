#!/usr/bin/env bash
set -euo pipefail

# Stage 2 of Task 3: train the Stage 1 checkpoint on SmolTalk only, then
# evaluate it on ARC-Easy, ARC-Challenge, and GSM8K.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT_ROOT="$PWD"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
MIDTRAIN_TAG="${MIDTRAIN_TAG:-d2_bpb}"
SFT_TAG="${SFT_TAG:-d2_bpb_sft}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_TOKENS="${EVAL_TOKENS:-1048576}"
CHAT_EVAL_BATCH_SIZE="${CHAT_EVAL_BATCH_SIZE:-8}"

export NANOCHAT_BASE_DIR="$PROJECT_ROOT/results/task2_models/vocab_${VOCAB_SIZE}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/results/task3_cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

MIDTRAIN_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MIDTRAIN_TAG"
SFT_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$SFT_TAG"
LOG_DIR="$PROJECT_ROOT/results/task3_sft/vocab_${VOCAB_SIZE}"
LOG_FILE="$LOG_DIR/sft.log"
EVAL_LOG_FILE="$LOG_DIR/stage2_eval.log"

if [[ ! -d "$MIDTRAIN_CHECKPOINT_DIR" ]]; then
    echo "Error: Stage 1 checkpoint directory not found: $MIDTRAIN_CHECKPOINT_DIR" >&2
    echo "Run CUDA_VISIBLE_DEVICES=<gpu> bash runs/task3_midtrain.sh first." >&2
    exit 1
fi

shopt -s nullglob
midtrain_models=("$MIDTRAIN_CHECKPOINT_DIR"/model_*.pt)
existing_sft_models=("$SFT_CHECKPOINT_DIR"/model_*.pt)
shopt -u nullglob

if (( ${#midtrain_models[@]} == 0 )); then
    echo "Error: no Stage 1 model checkpoint found in: $MIDTRAIN_CHECKPOINT_DIR" >&2
    exit 1
fi

if (( ${#existing_sft_models[@]} > 0 )); then
    echo "Error: a Stage 2 checkpoint already exists in: $SFT_CHECKPOINT_DIR" >&2
    echo "Choose a different SFT_TAG to preserve the existing checkpoint." >&2
    exit 1
fi

mkdir -p "$HF_HOME" "$LOG_DIR"

echo "============================================================"
echo "Task 3 Stage 2: SmolTalk-only supervised fine-tuning"
echo "Vocabulary size: $VOCAB_SIZE"
echo "Loading Stage 1 checkpoint: $MIDTRAIN_CHECKPOINT_DIR"
echo "Saving Stage 2 checkpoint: $SFT_CHECKPOINT_DIR"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "Training log: $LOG_FILE"
echo "============================================================"

python -m scripts.chat_sft \
    --stage=sft \
    --model-tag="$MIDTRAIN_TAG" \
    --output-tag="$SFT_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens="$EVAL_TOKENS" \
    --chatcore-every=-1 \
    --run=dummy \
    2>&1 | tee "$LOG_FILE"

echo
echo "Stage 2 complete. Checkpoint saved in: $SFT_CHECKPOINT_DIR"

echo
echo "============================================================"
echo "Evaluating Stage 2 checkpoint"
echo "Tasks: ARC-Easy, ARC-Challenge, GSM8K"
echo "Evaluation log: $EVAL_LOG_FILE"
echo "============================================================"

python -m scripts.chat_eval \
    --source=sft \
    --model-tag="$SFT_TAG" \
    --task-name='ARC-Easy|ARC-Challenge|GSM8K' \
    --batch-size="$CHAT_EVAL_BATCH_SIZE" \
    2>&1 | tee "$EVAL_LOG_FILE"

echo
echo "Stage 2 evaluation complete. Results saved in: $EVAL_LOG_FILE"
