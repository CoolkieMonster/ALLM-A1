#!/usr/bin/env python3
"""Task 2: train depth-2 base model, collect BPB curve, summary, and raw samples."""
import argparse, csv, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NC = ROOT/"nanochat-master"
OUT, LOG = ROOT/"results"/"task2", ROOT/"logs"/"task2"
BASE = Path(os.getenv("NANOCHAT_BASE_DIR", Path.home()/".cache"/"nanochat")).expanduser()
TAG = "d2"

def run(cmd, log, capture=False):
    print("$", " ".join(map(str, cmd)))
    if capture:
        p = subprocess.run(cmd, cwd=NC, text=True, capture_output=True, check=True)
        text = p.stdout + p.stderr; log.write_text(text); return text
    with log.open("w") as f:
        p = subprocess.Popen(cmd, cwd=NC, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout: print(line, end=""); f.write(line)
        if p.wait(): raise SystemExit(p.returncode)

def ckpts():
    return sorted((BASE/"base_checkpoints"/TAG).glob("model_*.pt"))

def train():
    run([sys.executable,"-m","scripts.base_train","--depth","2",
         "--device-batch-size","4","--eval-every","50","--save-every","50",
         "--core-metric-every","-1"], LOG/"pretrain.log")

def analyze(split_tokens):
    rows=[]
    for m in ckpts():
        step=int(m.stem.split("_")[1])
        text=run([sys.executable,"-m","scripts.base_eval","--eval","bpb",
                  "--model-tag",TAG,"--step",str(step),"--device-batch-size","4",
                  "--split-tokens",str(split_tokens)], LOG/f"bpb_{step:06d}.log", True)
        tr=float(re.search(r"train bpb:\s*([\d.]+)",text).group(1))
        va=float(re.search(r"val bpb:\s*([\d.]+)",text).group(1))
        rows.append((step,tr,va))
    with (OUT/"bpb.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["step","train_bpb","val_bpb"]); w.writerows(rows)

    import matplotlib.pyplot as plt
    x,tr,va=zip(*rows); plt.figure()
    plt.plot(x,tr,marker="o",label="train"); plt.plot(x,va,marker="o",label="validation")
    plt.xlabel("optimizer step"); plt.ylabel("bits per byte (BPB)"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT/"bpb_curve.pdf"); plt.close()

    log=(LOG/"pretrain.log").read_text()
    def n(p):
        m=re.search(p,log,re.M); return int(m.group(1).replace(",","")) if m else None
    total=n(r"^total\s*:\s*([\d,]+)")
    tm=n(r"^transformer_matrices\s*:\s*([\d,]+)")
    lm=n(r"^lm_head\s*:\s*([\d,]+)")
    tokens=n(r"Total number of training tokens:\s*([\d,]+)")
    scaling=tm+lm if tm is not None and lm is not None else None
    (OUT/"summary.txt").write_text(
        f"total_parameters={total}\nscaling_parameters={scaling}\n"
        f"nanochat_training_tokens={tokens}\n"
        f"chinchilla_approx_tokens_20x={20*scaling if scaling else None}\n"
        f"bpb_eval_tokens_per_split={split_tokens}\n")

def sample():
    step=int(ckpts()[-1].stem.split("_")[1])
    text=run([sys.executable,"-m","scripts.base_eval","--eval","sample",
              "--model-tag",TAG,"--step",str(step),"--device-batch-size","4"],
             LOG/"samples_full.log", True)
    (OUT/"raw_completions.txt").write_text(text); print(text)

def main():
    OUT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser(); p.add_argument("action",nargs="?",default="all",
        choices=["train","analyze","sample","all"])
    p.add_argument("--split-tokens",type=int,default=1_048_576)
    a=p.parse_args()
    if a.action in ("train","all"): train()
    if a.action in ("analyze","all"): analyze(a.split_tokens)
    if a.action in ("sample","all"): sample()

if __name__=="__main__": main()
