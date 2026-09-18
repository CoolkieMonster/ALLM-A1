#!/usr/bin/env bash
set -euo pipefail

# Stage 1 of Task 3: continue a pretrained base model on MMLU and GSM8K only.
# Run from any directory with:
#   CUDA_VISIBLE_DEVICES=0 bash runs/task3_midtrain.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT_ROOT="$PWD"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d2_bpb}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_TOKENS="${EVAL_TOKENS:-1048576}"
CHAT_EVAL_BATCH_SIZE="${CHAT_EVAL_BATCH_SIZE:-8}"
INSPECT_ROWS="${INSPECT_ROWS:-3}"
MMLU_EPOCHS="${MMLU_EPOCHS:-3}"
GSM8K_EPOCHS="${GSM8K_EPOCHS:-4}"

export NANOCHAT_BASE_DIR="$PROJECT_ROOT/results/task2_models/vocab_${VOCAB_SIZE}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/results/task3_cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

BASE_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$BASE_MODEL_TAG"
MIDTRAIN_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$BASE_MODEL_TAG"
LOG_DIR="$PROJECT_ROOT/results/task3_midtrain/vocab_${VOCAB_SIZE}"
LOG_FILE="$LOG_DIR/midtrain.log"
EVAL_LOG_FILE="$LOG_DIR/stage1_eval.log"
DATA_INSPECTION_LOG="$LOG_DIR/data_inspection.log"

if [[ ! -d "$BASE_CHECKPOINT_DIR" ]]; then
    echo "Error: base checkpoint directory not found: $BASE_CHECKPOINT_DIR" >&2
    exit 1
fi

shopt -s nullglob
base_models=("$BASE_CHECKPOINT_DIR"/model_*.pt)
existing_midtrain_models=("$MIDTRAIN_CHECKPOINT_DIR"/model_*.pt)
shopt -u nullglob

if (( ${#base_models[@]} == 0 )); then
    echo "Error: no base model checkpoint found in: $BASE_CHECKPOINT_DIR" >&2
    exit 1
fi

if (( ${#existing_midtrain_models[@]} > 0 )); then
    echo "Error: a mid-training checkpoint already exists in: $MIDTRAIN_CHECKPOINT_DIR" >&2
    echo "Choose a different BASE_MODEL_TAG or move the existing checkpoint first." >&2
    exit 1
fi

mkdir -p "$HF_HOME" "$LOG_DIR"

echo "============================================================"
echo "Task 3 dataset sanity check"
echo "Printing $INSPECT_ROWS MMLU rows and $INSPECT_ROWS SmolTalk rows"
echo "Inspection log: $DATA_INSPECTION_LOG"
echo "============================================================"

python -m scripts.inspect_task3_data \
    --num-rows="$INSPECT_ROWS" \
    --mmlu-epochs="$MMLU_EPOCHS" \
    2>&1 | tee "$DATA_INSPECTION_LOG"

echo

echo "============================================================"
echo "Task 3 Stage 1: MMLU + GSM8K mid-training"
echo "Vocabulary size: $VOCAB_SIZE"
echo "Loading base checkpoint: $BASE_CHECKPOINT_DIR"
echo "Saving mid-training checkpoint: $MIDTRAIN_CHECKPOINT_DIR"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "MMLU epochs: $MMLU_EPOCHS"
echo "GSM8K epochs: $GSM8K_EPOCHS"
echo "Log file: $LOG_FILE"
echo "============================================================"

python -m scripts.chat_sft \
    --stage=midtrain \
    --model-tag="$BASE_MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --mmlu-epochs="$MMLU_EPOCHS" \
    --gsm8k-epochs="$GSM8K_EPOCHS" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens="$EVAL_TOKENS" \
    --chatcore-every=-1 \
    --run=dummy \
    2>&1 | tee "$LOG_FILE"

echo
echo "Stage 1 complete. Checkpoint saved in: $MIDTRAIN_CHECKPOINT_DIR"

echo
echo "============================================================"
echo "Evaluating Stage 1 checkpoint"
echo "Tasks: ARC-Easy, ARC-Challenge, GSM8K"
echo "Evaluation log: $EVAL_LOG_FILE"
echo "============================================================"

python -m scripts.chat_eval \
    --source=sft \
    --model-tag="$BASE_MODEL_TAG" \
    --task-name='ARC-Easy|ARC-Challenge|GSM8K' \
    --batch-size="$CHAT_EVAL_BATCH_SIZE" \
    2>&1 | tee "$EVAL_LOG_FILE"

echo
echo "Stage 1 evaluation complete. Results saved in: $EVAL_LOG_FILE"
