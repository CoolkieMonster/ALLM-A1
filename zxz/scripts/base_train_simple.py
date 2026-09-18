"""A small, readable language-model training example.

This script intentionally leaves out the performance and distributed-training
features used by ``scripts.base_train``.  It keeps only the basic workflow:

    text -> tokenizer -> input/target batches -> model -> loss
         -> backward() -> AdamW step

Run it from the nanochat repository root, for example:

    python -m scripts.base_train_simple \
        --tokenizer-dir results/task1_tokenizers/tokenizer_8192 \
        --max-seq-len 128 --batch-size 4 --depth 2 \
        --n-embd 128 --n-head 4 --steps 100
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer


class SimpleGPT(nn.Module):
    """A small causal Transformer with no nanochat-specific optimizations."""

    def __init__(self, vocab_size, max_seq_len, n_embd, n_layer, n_head):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Convert token IDs and positions into vectors.
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)

        # A standard PyTorch Transformer stack.
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        # True entries are masked: a token cannot look at future tokens.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        logits = self.lm_head(self.final_norm(hidden))

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )
        return loss


def token_stream(tokenizer, split="train"):
    """Yield token IDs from the parquet dataset forever."""
    bos_id = tokenizer.get_bos_token_id()
    while True:
        for batch in parquets_iter_batched(split=split):
            for document in batch:
                # BOS separates documents. The target is made by shifting later.
                yield bos_id
                yield from tokenizer.encode(document)


def batch_stream(tokenizer, batch_size, seq_len, split="train"):
    """Pack the token stream into [batch_size, seq_len] input/target tensors."""
    stream = token_stream(tokenizer, split=split)
    tokens = []
    needed = batch_size * (seq_len + 1)

    while True:
        while len(tokens) < needed:
            tokens.extend(next(stream) for _ in range(needed - len(tokens)))

        row = torch.tensor(tokens[:needed], dtype=torch.long)
        tokens = tokens[needed:]
        row = row.view(batch_size, seq_len + 1)
        yield row[:, :-1], row[:, 1:]


def choose_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Readable single-device GPT training")
    parser.add_argument("--tokenizer-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="simple_model.pt")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = choose_device(args.device)
    tokenizer = RustBPETokenizer.from_directory(args.tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()

    model = SimpleGPT(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        n_embd=args.n_embd,
        n_layer=args.depth,
        n_head=args.n_head,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_batches = batch_stream(
        tokenizer,
        batch_size=args.batch_size,
        seq_len=args.max_seq_len,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"vocab_size: {vocab_size:,}")
    print(f"parameters: {parameter_count:,}")

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = next(train_batches)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # The essential training loop.
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % args.print_every == 0:
            print(f"step {step:04d}/{args.steps:04d} | loss {loss.item():.4f}")

    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = Path(get_base_dir()) / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "vocab_size": vocab_size,
        },
        save_path,
    )
    print(f"saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
