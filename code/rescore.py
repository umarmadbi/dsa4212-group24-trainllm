"""
Re-score a saved final model (outside the training budget; DECLARATIONS section 5 protocol).

Plain language: every run saves its final model (`final_model.pt` on that workspace's runs Volume).
This tool loads it and computes bits per byte again, on any validation set, optionally with some
documents left out. Uses:
* E03 rule R3 (D25): AVG4 with contaminated validation documents removed (the SAME documents for every
  arm, from results/contamination.json), so a near-duplicate cannot favour one training corpus;
* N26 seed 0 (the reused E01 horizon run) was scored on FineWeb-Edu only: give it the other three sets;
* a reproducibility check: with nothing removed, the score must equal the one written at the end of
  training (up to GPU non-determinism, ~1e-5 BPB). Every call reports that difference.

Scoring is exactly `bpb.evaluate_bpb` (same windows, same EOT rule). Removing documents rebuilds the file
in the same format (each kept document followed by EOT, original order), so windows shift; that is
inherent to scoring a different document set and identical for every model scored with the same list.

Must run in the Modal workspace that holds the checkpoint:
    modal run modal_app.py::rescore --run-ids "E03-N49_dclm__s0,E03-N49_dclm__s1" --exclude contamination
    modal run modal_app.py::rescore --run-ids "E01-N26_lr4e-3_60min__s0" --sets "fineweb,dclm,wikipedia"
    modal run modal_app.py::rescore --run-ids "E07-V4096_lr2e-3_60min__s0" --sets fineweb_edu --stride 128
(E07 diagnostic, D36: overlapping windows; vocabulary-variant models are scored on their own variant's files.)
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import torch

from bpb import evaluate_bpb, evaluate_bpb_sliding
from model import GPT, GPTConfig


def rebuild_without(tokens: np.ndarray, eot: int, exclude) -> tuple[np.ndarray, int, int]:
    """Evaluation file with the documents whose (0-based, file-order) index is in `exclude` removed.
    Returns (tokens, documents kept, documents removed)."""
    exclude = set(int(i) for i in exclude)
    ends = np.flatnonzero(tokens == eot)
    parts, start, kept, removed = [], 0, 0, 0
    for i, e in enumerate(ends):
        if i in exclude:
            removed += 1
        else:
            parts.append(tokens[start:e + 1]); kept += 1
        start = e + 1
    if start < len(tokens):
        parts.append(tokens[start:])
    out = np.concatenate(parts) if parts else tokens[:0]
    return out.astype(tokens.dtype), kept, removed


def load_model(path: Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    m = GPT(GPTConfig(n_layer=c["n_layer"], n_head=c["n_head"], n_embd=c["n_embd"],
                      block_size=c["block_size"], vocab_size=c["vocab_size"],
                      dropout=0.0, bias=c["bias"]))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), c, ck.get("seed")


def data_dir_for(c: dict, data_dir) -> Path:
    """The data a model must be scored on: its vocabulary variant's files if it was trained on one (D36)."""
    data_dir = Path(data_dir)
    return data_dir / "vocab" / c["data_variant"] if c.get("data_variant") else data_dir


def rescore(run_dir: str, data_dir: str, sets, exclude: dict | None = None, device: str | None = None,
            stride: int = 0, split: str = "val") -> dict:
    """`stride` > 0 (E07 diagnostic, D36): also score with overlapping windows (bpb.evaluate_bpb_sliding).
    A model trained on a vocabulary variant (config data_variant) is scored on that variant's files.
    `split` (D42): "val" (default; everything as before) or "test". Test scoring is only ever called through
    code/evaluate_test.py, which enforces the test lock (DECLARATIONS sec. 5); with "test" there is no stored
    score to compare with (training never touches the test split)."""
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    model, c, seed = load_model(run_dir / "final_model.pt", device)
    data_dir = data_dir_for(c, data_dir)
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" and c.get("dtype") == "bfloat16" else contextlib.nullcontext())
    meta = json.loads((data_dir / "tokenizer" / "meta.json").read_text())
    tb = np.load(data_dir / "tokenizer" / "token_bytes.npy")
    stored = {}
    rj = run_dir / "result.json"
    if rj.exists() and split == "val":
        stored = {k: v.get("bpb") for k, v in (json.loads(rj.read_text()).get("final_val") or {}).items()}
    if meta["vocab_size"] != c["vocab_size"]:
        raise ValueError(f"tokenizer vocab {meta['vocab_size']} != model vocab {c['vocab_size']}")
    out = dict(run_id=run_dir.name, seed=seed, device=device, data_dir=str(data_dir), split=split, sets={})
    for s in sets:
        toks = np.fromfile(data_dir / "tokens" / s / f"{split}.bin", dtype=np.uint16)
        full = evaluate_bpb(model, toks, tb, c["block_size"], c.get("eval_batch", 32), meta["eot_id"], device, ctx)
        e = dict(full=full, stored_bpb=stored.get(s),
                 diff_vs_stored=(full["bpb"] - stored[s]) if stored.get(s) is not None else None)
        ex = (exclude or {}).get(s)
        if ex:
            t2, kept, removed = rebuild_without(toks, meta["eot_id"], ex)
            r2 = evaluate_bpb(model, t2, tb, c["block_size"], c.get("eval_batch", 32), meta["eot_id"], device, ctx)
            e["excluded"] = dict(r2, docs_kept=kept, docs_removed=removed)
        if stride:
            e["sliding"] = evaluate_bpb_sliding(model, toks, tb, c["block_size"], stride, c.get("eval_batch", 32),
                                                meta["eot_id"], device, ctx)
        out["sets"][s] = e
    return out
