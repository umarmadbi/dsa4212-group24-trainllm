"""
Deterministic bits-per-byte evaluation (DECLARATIONS.md section 5). `evaluate_bpb` is the declared metric;
`evaluate_bpb_sliding` (added D36) is an E07 diagnostic only.

    BPB = sum over scored targets of -log p(target | context)  /  (bytes * ln 2)

Scoring protocol, identical for every run:
* The token file is already documents joined with <|endoftext|> (EOT) after each one.
* It is cut into non-overlapping windows of `T` tokens. Window i predicts tokens
  [i*T+1, i*T+T] from tokens [i*T, i*T+T-1], so every target is scored exactly once.
  The first token of the file and any tail shorter than a window are not scored.
* EOT targets contribute neither loss nor bytes.
* `bytes` is the sum of the UTF-8 byte lengths of the scored target tokens
  (token_bytes.npy). Because the tokenizer is byte-level and lossless, this equals
  the byte count of the text those tokens represent (checked in prepare_data.py).
* Sums are accumulated in float64 and divided once at the end (no averaging of
  per-window ratios).
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_bpb(model, tokens: np.ndarray, token_bytes: np.ndarray, T: int, batch: int,
                 eot_id: int, device: str, ctx=None, max_windows: int | None = None) -> dict:
    ctx = ctx or contextlib.nullcontext()
    was_training = model.training
    model.eval()
    n_windows = (len(tokens) - 1) // T
    if max_windows:
        n_windows = min(n_windows, max_windows)
    offsets = np.arange(T + 1)
    tb = torch.as_tensor(token_bytes.astype(np.int64), device=device)
    sum_nll = 0.0
    sum_bytes = 0
    n_scored = 0
    n_eot = 0
    for start in range(0, n_windows, batch):
        idx = np.arange(start, min(start + batch, n_windows), dtype=np.int64) * T
        w = torch.from_numpy(tokens[idx[:, None] + offsets].astype(np.int64)).to(device)
        x, y = w[:, :-1], w[:, 1:]
        with ctx:
            logits, _ = model(x)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                              reduction="none")
        mask = y.reshape(-1) != eot_id
        sum_nll += nll[mask].double().sum().item()
        sum_bytes += int(tb[y.reshape(-1)[mask]].sum().item())
        n_scored += int(mask.sum().item())
        n_eot += int((~mask).sum().item())
    if was_training:
        model.train()
    return dict(bpb=sum_nll / (sum_bytes * math.log(2)),
                loss_per_token=sum_nll / n_scored,
                scored_tokens=n_scored, scored_bytes=sum_bytes, eot_targets_excluded=n_eot,
                bytes_per_token=sum_bytes / n_scored, windows=n_windows, context=T)


@torch.no_grad()
def evaluate_bpb_sliding(model, tokens: np.ndarray, token_bytes: np.ndarray, T: int, stride: int, batch: int,
                         eot_id: int, device: str, ctx=None) -> dict:
    """DIAGNOSTIC ONLY (E07, D36), never the declared metric: bits per byte with overlapping windows.

    Windows of T tokens start every `stride` tokens. The first window scores all its T targets; every later
    window scores only its last `stride` targets, so every target is scored exactly once (as in evaluate_bpb)
    but, apart from the first T, each one sees at least T - stride tokens of context instead of 0..T-1.
    Purpose: the declared protocol's non-overlapping windows reset the context every T TOKENS, i.e. more often
    per byte for a small vocabulary. Comparing the vocabulary gap under both protocols shows how much of it is
    this evaluation effect. With stride == T this is exactly evaluate_bpb (tested). The tail shorter than
    `stride` is not scored."""
    if not 0 < stride <= T:
        raise ValueError(f"stride must be in 1..{T}, got {stride}")
    ctx = ctx or contextlib.nullcontext()
    was_training = model.training
    model.eval()
    n_windows = (len(tokens) - 1 - T) // stride + 1 if len(tokens) - 1 >= T else 0
    offsets = np.arange(T + 1)
    tb = torch.as_tensor(token_bytes.astype(np.int64), device=device)
    keep = torch.zeros(T, dtype=torch.bool, device=device)
    keep[T - stride:] = True
    sum_nll, sum_bytes, n_scored, n_eot = 0.0, 0, 0, 0
    for start in range(0, n_windows, batch):
        ks = np.arange(start, min(start + batch, n_windows), dtype=np.int64)
        idx = ks * stride
        w = torch.from_numpy(tokens[idx[:, None] + offsets].astype(np.int64)).to(device)
        x, y = w[:, :-1], w[:, 1:]
        with ctx:
            logits, _ = model(x)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                              reduction="none").reshape(len(ks), T)
        pos = keep.expand(len(ks), T).clone()             # target positions this window scores
        if ks[0] == 0:
            pos[0] = True                                  # first window: every target
        is_eot = y == eot_id
        sel = pos & ~is_eot
        sum_nll += nll[sel].double().sum().item()
        sum_bytes += int(tb[y[sel]].sum().item())
        n_scored += int(sel.sum().item())
        n_eot += int((pos & is_eot).sum().item())
    if was_training:
        model.train()
    return dict(bpb=sum_nll / (sum_bytes * math.log(2)), loss_per_token=sum_nll / n_scored,
                scored_tokens=n_scored, scored_bytes=sum_bytes, eot_targets_excluded=n_eot,
                bytes_per_token=sum_bytes / n_scored, windows=n_windows, context=T, stride=stride)
