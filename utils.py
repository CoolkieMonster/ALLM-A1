#!/usr/bin/env python3
"""Small helpers shared by the assignment task scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NANOCHAT = ROOT / "nanochat-master"


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def run_logged(cmd, log_path: Path, base_dir: Path | None = None, capture: bool = False) -> str:
    """Run a command, mirror stdout to a log, and raise on failure."""
    env = os.environ.copy()
    if base_dir is not None:
        env["NANOCHAT_BASE_DIR"] = str(base_dir)
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(map(str, cmd)))

    if capture:
        done = subprocess.run(
            cmd, cwd=NANOCHAT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        text = done.stdout or ""
        log_path.write_text(text, encoding="utf-8")
        print(text, end="")
        if done.returncode != 0:
            raise subprocess.CalledProcessError(done.returncode, cmd)
        return text

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd, cwd=NANOCHAT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)
    return ""


def replace_symlink(link: Path, target: Path) -> None:
    """Create or replace a directory symlink."""
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target:
            return
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            shutil.rmtree(link)
    link.symlink_to(target, target_is_directory=True)


def available_tokenizers(checkpoint_dir: Path, expected=(8192, 32768)) -> list[int]:
    """Return usable tokenizer sizes, warning about missing expected ones."""
    found = []
    for vocab_size in expected:
        path = checkpoint_dir / f"tokenizer_{vocab_size}" / "tokenizer.pkl"
        if path.is_file():
            found.append(vocab_size)
        else:
            warn(f"tokenizer {vocab_size:,} not found; skipping it")
    return found


def latest_checkpoint_step(checkpoint_dir: Path) -> int | None:
    """Return the largest model_<step>.pt step, or None if no checkpoint exists."""
    steps = []
    for path in checkpoint_dir.glob("model_*.pt"):
        try:
            steps.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(steps) if steps else None


def available_task2_runs(checkpoint_root: Path, model_tag: str = "d2") -> list[tuple[int, Path]]:
    """Return Task-2 runs that contain both a tokenizer and a base checkpoint."""
    runs = []
    for run_dir in sorted(checkpoint_root.glob("vocab_*")):
        try:
            vocab_size = int(run_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        tokenizer_ok = (run_dir / "tokenizer" / "tokenizer.pkl").is_file()
        model_dir = run_dir / "base_checkpoints" / model_tag
        if tokenizer_ok and latest_checkpoint_step(model_dir) is not None:
            runs.append((vocab_size, run_dir))
    return runs


def select_task2_run(checkpoint_root: Path, preferred_vocab: int = 32768, model_tag: str = "d2") -> tuple[int, Path]:
    """Prefer the 32K model, otherwise fall back to another complete Task-2 run."""
    runs = available_task2_runs(checkpoint_root, model_tag)
    if not runs:
        raise FileNotFoundError("No complete Task-2 model run was found. Run task2.py first.")
    for vocab_size, run_dir in runs:
        if vocab_size == preferred_vocab:
            return vocab_size, run_dir
    vocab_size, run_dir = max(runs, key=lambda item: item[0])
    warn(f"preferred {preferred_vocab:,} Task-2 model is unavailable; using {vocab_size:,} instead")
    return vocab_size, run_dir


def reset_dir(path: Path) -> None:
    """Remove an old generated directory and recreate it."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def python_cmd(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", module, *map(str, args)]
