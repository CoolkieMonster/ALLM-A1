# ALLM Assignment 1 - Reproduction Guide

The project layout:

```text
ALLM-A1/
├── README.md
├── task1.py
├── task2.py
├── task3.py
├── task4.py
├── nanochat-master/
│   └── scripts/
│       ├── chat_sft.py
│       └── chat_sft_staged.py   # assignment modification
...below the folders are outputs...
├── data/                        # data/checkpoints, controlled by `NANOCHAT_BASE_DIR`
├── logs/                        # raw experiment logs, fixed in task codes
└── results/                     # report-ready artifacts, fixed in task codes
```

`data/` does not need to be committed. Its location is controlled by `NANOCHAT_BASE_DIR`, so another user can store nanochat data anywhere on their machine.

## 1. Setup

Clone nanochat into the expected directory:

```bash
git clone https://github.com/karpathy/nanochat.git nanochat-master
cd nanochat-master
uv sync --extra gpu --group dev
source .venv/bin/activate
cd ..
```

Place the supplied assignment files as follows:

```text
ALLM-A1/task1.py
ALLM-A1/task2.py
ALLM-A1/task3.py
ALLM-A1/nanochat-master/scripts/chat_sft_staged.py
```

Choose a nanochat data directory. For example:

```bash
export NANOCHAT_BASE_DIR=/local/$USER/ALLM-A1/data
```

Any writable path is valid. If this variable is not set, nanochat uses its normal cache location under `~/.cache/nanochat`.


The scripts create `logs/taskX/` and `results/taskX/` automatically.

---

# Task 1 — Tokenization

Task 1 trains two Rust BPE tokenizers on the CLIMBMix sample, preserves both tokenizers, and produces evidence for the tokenizer analysis.

Run the full task:

```bash
python task1.py all --merge-example "YOUR OWN ORIGINAL EXAMPLE"
```

The merge example must be your own example for the report.

| Sub-task | Output | Saved at |
|---|---|---|
| Download CLIMBMix sample | CLIMBMix parquet shards | `$NANOCHAT_BASE_DIR/base_data_climbmix/` |
| Train 8,192-token BPE | tokenizer files | `$NANOCHAT_BASE_DIR/task1_tokenizers/vocab_8192/` |
| Train 32,768-token BPE | tokenizer files | `$NANOCHAT_BASE_DIR/task1_tokenizers/vocab_32768/` |
| Keep downstream tokenizer | active 32,768 tokenizer | `$NANOCHAT_BASE_DIR/tokenizer/` |
| Record training output | dataset + tokenizer logs | `logs/task1/` |
| Compare compression | token counts, tokens/character, bytes/token | `results/task1/tokenizer_metrics.csv` |
| Inspect artifacts | English, numbers, code, non-English tokenization | `results/task1/tokenizer_examples.txt` |
| Show BPE merge example | learned merge trace and counts for the chosen example | `results/task1/merge_trace.txt` |


To separate training and analysis:

```bash
python task1.py train
python task1.py analyze --merge-example "YOUR OWN ORIGINAL EXAMPLE"
```

---

# Task 2 — Pre-training

Task 2 trains a depth-2 base model, saves regular checkpoints, evaluates train/validation BPB at those checkpoints, plots the BPB curves, extracts basic scaling information, and samples the final pretrained model.

Run everything:

```bash
python task2.py all
```

| Sub-task | Output | Saved at |
|---|---|---|
| Depth-2 pretraining | base checkpoints | `$NANOCHAT_BASE_DIR/base_checkpoints/d2/` |
| Raw training record | training loss/configuration/output | `logs/task2/pretrain.log` |
| BPB evaluation per checkpoint | raw evaluator logs | `logs/task2/bpb_*.log` |
| Train/validation BPB data | curve values | `results/task2/bpb.csv` |
| BPB plot | train + validation curve | `results/task2/bpb_curve.pdf` |
| Architecture/scaling summary | parameters, training tokens, Chinchilla estimate | `results/task2/summary.txt` |
| Raw pretrained sampling | base-model completions | `results/task2/raw_completions.txt` |
| Raw sampling log | complete sampling command output | `logs/task2/samples_full.log` |


The default BPB estimate uses 1,048,576 tokens per train/validation split at each saved checkpoint. It can be changed with:

```bash
python task2.py all --split-tokens 4194304
```

Individual stages are also available:

```bash
python task2.py train
python task2.py analyze
python task2.py sample
```

---

# Task 3 — Mid-training and SFT

Task 3 introduces the assignment-required extra training stage. The supplied `chat_sft_staged.py` is a modified version of nanochat's original `scripts/chat_sft.py`.

Stage 1 starts from the pretrained base checkpoint and trains only on MMLU + GSM8K. Stage 2 starts from the Stage-1 checkpoint and trains only on SmolTalk.

Run the complete pipeline:

```bash
python task3.py all
```

| Sub-task | Output | Saved at |
|---|---|---|
| Inspect training datasets | sizes + first examples from MMLU, GSM8K, SmolTalk | `results/task3/dataset_examples.txt` |
| Evaluate pretrained base | ARC-Easy, ARC-Challenge, GSM8K raw results | `logs/task3/base_eval.log` |
| Stage 1: mid-training | MMLU + GSM8K checkpoint | `$NANOCHAT_BASE_DIR/chatsft_checkpoints/midtrain-d2/` |
| Evaluate Stage 1 | ARC-Easy, ARC-Challenge, GSM8K | `logs/task3/midtrain_eval.log` |
| Stage 2: SFT | SmolTalk checkpoint | `$NANOCHAT_BASE_DIR/chatsft_checkpoints/sft-d2/` |
| Evaluate Stage 2 | ARC-Easy, ARC-Challenge, GSM8K | `logs/task3/sft_eval.log` |
| Consolidate comparison | base vs midtrain vs SFT benchmark table | `results/task3/benchmark_scores.csv` |

Before Task 3, Task 2 must have produced the base checkpoint:

```text
$NANOCHAT_BASE_DIR/base_checkpoints/d2/
```

`all` performs:

```text
dataset inspection
→ base evaluation
→ Stage-1 mid-training
→ full Stage-1 evaluation
→ Stage-2 SFT
→ full Stage-2 evaluation
→ benchmark table
```

For easier debugging or restarting, the same single script can be run stage by stage:

```bash
python task3.py inspect
python task3.py midtrain
python task3.py sft
python task3.py eval
```

`midtrain` trains Stage 1 and then evaluates it. `sft` trains Stage 2 and then evaluates it. `eval` re-evaluates all three existing checkpoints without retraining.

---

