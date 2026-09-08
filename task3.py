#!/usr/bin/env python3
"""Task 3: inspect data, run mid-training + SFT, and evaluate all three model stages."""
import argparse, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NC=ROOT/"nanochat-master"
OUT,LOG=ROOT/"results"/"task3",ROOT/"logs"/"task3"
TASKS="ARC-Easy|ARC-Challenge|GSM8K"

def run(cmd,log):
    print("$"," ".join(map(str,cmd)))
    with log.open("w") as f:
        p=subprocess.Popen(cmd,cwd=NC,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        for line in p.stdout: print(line,end=""); f.write(line)
        if p.wait(): raise SystemExit(p.returncode)

def inspect():
    sys.path.insert(0,str(NC))
    from tasks.mmlu import MMLU
    from tasks.gsm8k import GSM8K
    from tasks.smoltalk import SmolTalk
    sets={"MMLU":MMLU(subset="all",split="auxiliary_train"),
          "GSM8K":GSM8K(subset="main",split="train"),
          "SmolTalk":SmolTalk(split="train")}
    with (OUT/"dataset_examples.txt").open("w") as f:
        for name,ds in sets.items():
            print(f"\n{name}: {len(ds):,} examples",file=f)
            for i in range(min(3,len(ds))): print(f"[{i}] {ds[i]}",file=f)

def midtrain():
    run([sys.executable,"-m","scripts.chat_sft_staged",
         "--stage","midtrain","--input-tag","d2","--output-tag","midtrain-d2",
         "--device-batch-size","4","--run","dummy"],LOG/"midtrain.log")

def sft():
    run([sys.executable,"-m","scripts.chat_sft_staged",
         "--stage","sft","--input-tag","midtrain-d2","--output-tag","sft-d2",
         "--device-batch-size","4","--run","dummy"],LOG/"sft.log")

def evaluate(source,tag,name):
    run([sys.executable,"-m","scripts.chat_eval","-i",source,"-g",tag,
         "-a",TASKS,"-b","4"],LOG/f"{name}_eval.log")

def table():
    import csv, re
    tasks=["ARC-Easy","ARC-Challenge","GSM8K"]
    rows=[]
    for stage in ["base","midtrain","sft"]:
        text=(LOG/f"{stage}_eval.log").read_text()
        scores={m.group(1):float(m.group(2)) for m in
                re.finditer(r"^(ARC-Easy|ARC-Challenge|GSM8K) accuracy:\s*([\\d.]+)%",text,re.M)}
        rows.append([stage]+[scores.get(t) for t in tasks])
    with (OUT/"benchmark_scores.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["stage"]+tasks); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser(); p.add_argument("action",nargs="?",default="all",
        choices=["inspect","midtrain","sft","eval","all"]); a=p.parse_args()
    if a.action in ("inspect","all"): inspect()
    if a.action=="all": evaluate("base","d2","base")
    if a.action in ("midtrain","all"): midtrain(); evaluate("sft","midtrain-d2","midtrain")
    if a.action in ("sft","all"): sft(); evaluate("sft","sft-d2","sft")
    if a.action=="eval":
        evaluate("base","d2","base")
        evaluate("sft","midtrain-d2","midtrain")
        evaluate("sft","sft-d2","sft")
    if a.action in ("eval","all"): table()

if __name__=="__main__": main()
