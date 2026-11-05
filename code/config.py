"""
Run configuration: the dataclass every config file fills in, and a strict loader.

Kept separate from train.py (no torch import) so that `modal_app.py::launch` can
validate every config on the local computer BEFORE any GPU job starts.

Why the loader is strict: PyYAML follows YAML 1.1, where `1.0e9` (no sign in the
exponent) is read as the STRING "1.0e9", while `6.0e-4` is a float. A string budget
or interval crashes the run on the GPU (bug found in E00 benchmarks, 2026-11-01).
The loader therefore converts each value to its declared type and rejects anything
that does not convert cleanly, and rejects unknown keys (typos).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    run_id: str = "debug"
    experiment: str = "E00"
    # data
    corpus: str = "fineweb_edu"
    train_tokens_limit: int = 0            # >0: use only the FIRST N train tokens (file order; not used for results)
    # >0: train on a RANDOM nested subset of ~N tokens (repetition experiment E04, D23):
    # whole 513-token windows drawn by a fixed permutation (key SUBSET_KEY in train.py, the
    # same for every run and seed), so every smaller subset is contained in every larger one
    # and each subset is a uniform sample of the whole train split.
    train_subset_tokens: int = 0
    # E04 follow-up F3 (D26): with a subset, move the window boundaries every pass (ShiftedSubsetLoader)
    train_subset_shift: bool = False
    # E03 follow-up R8 (D26): mix corpora by token share, e.g. ["fineweb_edu=0.6", "dclm=0.4"];
    # needs corpus: mix. Each corpus contributes a random pool of mix_pool_tokens (nested-subset key).
    corpus_mix: list = field(default_factory=list)
    mix_pool_tokens: int = 1_000_000_000
    # E05 (D28): "adamw" (declared baseline) or "muon" (torch.optim.Muon on the hidden 2-D matrices of the
    # transformer blocks; embeddings and LayerNorm weights stay on AdamW with lr/betas/weight_decay above)
    optimizer: str = "adamw"
    muon_lr: float = 0.0                   # peak Muon LR (same warmup/cosine schedule); required for muon
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_adjust_lr: str = "match_rms_adamw"   # update RMS matched to AdamW (Liu et al. 2025, "Muon is Scalable")
    eval_corpora: list = field(default_factory=lambda: ["fineweb_edu"])  # full val at the end
    curve_corpus: str = "fineweb_edu"      # fixed subset scored during training
    curve_tokens: int = 262_144            # ~1 MB of FineWeb-Edu text at 3.87 bytes/token
    # model (nanoGPT GPTConfig)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 512
    vocab_size: int = 8192
    dropout: float = 0.0
    bias: bool = False
    # batch
    micro_batch: int = 32
    grad_accum: int = 1
    # optimiser (AdamW, nanoGPT parameter groups)
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # budget and schedule
    budget_s: float = 3600.0
    schedule: str = "time"                 # "time" (declared) | "steps" (tests only, needs max_steps)
    max_steps: int = 0                     # 0 = no limit; tests and smoke runs only
    eval_every_frac: float = 0.05          # learning-curve points: every 5 % of the budget
    ckpt_every_s: float = 600.0
    ckpt_every_steps: int = 0              # tests only
    log_every_s: float = 30.0
    # Simulated preemption (Modal kill-and-resume test only; 0 = off). The container is
    # killed at this step on the FIRST attempt only, so the automatic retry resumes.
    test_crash_once_at_step: int = 0
    # system
    compile: bool = True
    dtype: str = "bfloat16"                # "bfloat16" | "float32"
    eval_batch: int = 32


def _coerce(name: str, typ: str, value):
    if typ == "float":
        if isinstance(value, bool):
            raise TypeError(f"{name}: expected a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise TypeError(f"{name}: expected a number, got {value!r}") from None
    if typ == "int":
        if isinstance(value, bool):
            raise TypeError(f"{name}: expected an integer, got {value!r}")
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise TypeError(f"{name}: expected an integer, got {value!r}") from None
        if f != int(f):
            raise TypeError(f"{name}: expected an integer, got {value!r}")
        return int(f)
    if typ == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"{name}: expected true/false, got {value!r}")
        return value
    if typ == "str":
        if not isinstance(value, str):
            raise TypeError(f"{name}: expected text, got {value!r}")
        return value
    if typ == "list":
        if isinstance(value, str):
            return [value]
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{name}: expected a list, got {value!r}")
        return list(value)
    return value


def load_config(path: str | None, overrides: dict | None = None) -> Config:
    d = {}
    if path:
        import yaml
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    d.update({k: v for k, v in (overrides or {}).items() if v is not None})
    fields = {f.name: f.type for f in dataclasses.fields(Config)}
    unknown = set(d) - set(fields)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = Config(**{k: _coerce(k, fields[k], v) for k, v in d.items()})
    if cfg.n_embd % cfg.n_head:
        raise ValueError("n_embd must be divisible by n_head")
    if cfg.schedule not in ("time", "steps"):
        raise ValueError(f"schedule must be 'time' or 'steps', got {cfg.schedule!r}")
    if cfg.schedule == "steps" and not cfg.max_steps:
        raise ValueError("schedule 'steps' needs max_steps")
    if cfg.optimizer not in ("adamw", "muon"):
        raise ValueError(f"optimizer must be 'adamw' or 'muon', got {cfg.optimizer!r}")
    if cfg.optimizer == "muon" and cfg.muon_lr <= 0:
        raise ValueError("optimizer 'muon' needs muon_lr > 0")
    if cfg.dtype not in ("bfloat16", "float32"):
        raise ValueError(f"dtype must be bfloat16 or float32, got {cfg.dtype!r}")
    return cfg
