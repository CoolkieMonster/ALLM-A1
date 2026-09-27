# ALLM Assignment 1 experiments

This repository keeps the assignment runners at the project root and uses the bundled `nanochat-master` as the training pipeline. Run the task scripts from the `ALLM-A1` root. Generated data, logs and results stay inside the project; the task scripts set `NANOCHAT_BASE_DIR` explicitly for better management.

## Structure

```text
ALLM-A1/
├── README.md
├── nanochat_changes.txt       # summary of modifications to upstream nanochat
├── utils.py
├── task1_train.py
├── task1_report.py
├── task2.py
├── task3.py
├── task4.py
├── nanochat-master/
│   ├── nanochat/
│   │   └── loss_eval.py       # modified: shared BPB statistics
│   └── scripts/
│       ├── base_train.py      # modified: online train BPB
│       └── chat_sft.py        # modified: separate mid-training and SFT
├── data/                      # generated/downloaded
├── logs/                      # generated raw experiment logs
└── results/                   # generated checkpoints and report artifacts
```

Changes inside original nanochat files are marked with `####[MODIFY]`. See `nanochat_changes.txt` for a short description and original/new line references.

## Environment setup

From the project root:

```bash
cd nanochat-master
uv sync --extra gpu --group dev
source .venv/bin/activate
cd ..
```

For CPU/MPS instead of CUDA, use:

```bash
cd nanochat-master
uv sync --extra cpu --group dev
source .venv/bin/activate
cd ..
```

If the environment already exists, only activation is needed:

```bash
cd nanochat-master
source .venv/bin/activate
cd ..
```

All commands below are run from the project root while this environment is active.

## Reproduce the assignment

Run the tasks in order:

```bash
python task1_train.py
python task1_report.py
python task2.py
python task3.py
python task4.py
```

### Task 1

`task1_train.py` checks/downloads CLIMBMix and trains the 8K and 32K BPE tokenizers on the same 500 MB sample. `task1_report.py` produces the compression, failure-case and merge-trace evidence. If one tokenizer is missing, the report continues with the available tokenizer and prints a warning.

### Task 2

`task2.py` trains a depth-2 model for each available Task-1 tokenizer. The modified `base_train.py` reports BPB from the actual optimizer-batch training data while validation BPB is measured on a fixed held-out sample. Raw train BPB is kept in the logs/CSV; the report plot applies an EMA (`beta=0.9`) and samples the train curve only at validation steps, so the two plotted curves use aligned optimizer steps.

The script also writes the scaling-law summary and five raw pre-trained completions. Missing tokenizer alternatives are skipped instead of stopping the whole task.

To enable W&B, change `WANDB_RUN_PREFIX = "dummy"` near the top of `task2.py`.

### Task 3

`task3.py` prefers the 32K Task-2 model and falls back to another complete Task-2 run if necessary. It then runs, in order:

1. dataset inspection and pretrained benchmark evaluation;
2. MMLU + GSM8K mid-training and evaluation;
3. SmolTalk SFT from the mid-training checkpoint and final evaluation.

The modified `scripts/chat_sft.py` contains both stages. Benchmark evaluation uses temperature `0.0`.

### Task 4

`task4.py` uses the final SFT checkpoint when available, then falls back to the mid-trained or pretrained model. It runs the five fixed prompts at temperatures `0.1`, `0.7` and `1.5`. Each prompt is saved as its own text file under `results/task4/report/`.

## Notes

The root task scripts intentionally have no subcommands. Important experiment settings such as device batch size and evaluation interval are grouped near the top of each file. Re-running a task overwrites the corresponding generated experiment outputs where appropriate.
