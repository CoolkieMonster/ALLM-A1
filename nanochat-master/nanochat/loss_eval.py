"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

# ####[MODIFY] Shared BPB statistics for online train logging.
@torch.no_grad()
def bpb_stats(losses, targets, token_bytes):
    """Return summed NLL (nats) and represented target bytes."""
    losses = losses.reshape(-1)
    targets = targets.reshape(-1)
    valid = targets >= 0
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    num_bytes = torch.where(
        valid,
        token_bytes[safe_targets],
        torch.zeros_like(targets, dtype=token_bytes.dtype),
    )
    total_nats = (losses * (num_bytes > 0)).sum(dtype=torch.float32)
    total_bytes = num_bytes.sum(dtype=torch.int64)
    return total_nats, total_bytes


@torch.no_grad()
def finalize_bpb(total_nats, total_bytes):
    """Reduce BPB statistics across ranks and return bits per byte."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)
# ####[MODIFY] End shared BPB statistics.


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        # ####[MODIFY] Reuse the same byte accounting as online train BPB.
        batch_nats, batch_bytes = bpb_stats(loss2d, y, token_bytes)
        total_nats += batch_nats
        total_bytes += batch_bytes
        # ####[MODIFY] End shared byte accounting.
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
