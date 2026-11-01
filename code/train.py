"""
Training script for every run in the project.

Based on nanoGPT train.py (commit 3adf61e), rewritten around the rules in
DECLARATIONS.md. Project modifications (each one a declared rule):

* Budget: a counted-time clock (section 1). It starts just before the first training
  step, synchronises the GPU before every reading, pauses during periodic validation
  and checkpointing, and stops training at the first step boundary at or after
  `budget_s`. torch.compile warm-up happens inside the first step, so it is counted.
* Learning-rate schedule: linear warmup over `warmup_steps`, then cosine decay to
  `min_lr_ratio * lr` driven by the fraction of the budget consumed, so every run
  finishes its decay exactly at the cap whatever its speed.
* Data order: a shuffled pass over non-overlapping windows. Every window is seen once
  per pass and each pass is reshuffled with (seed, pass). So "passes over the data"
  is exact, which the repetition experiment needs (nanoGPT samples random windows
  with replacement instead).
* Checkpoints: model, optimizer, step, counted time, tokens, data position, RNG states
  and logs, about every `ckpt_every_s` counted seconds. A restarted run resumes from
  the checkpoint with its counted time carried over (resuming never resets the budget).
* Evaluation: deterministic bits-per-byte (code/bpb.py) on a fixed ~1 MB validation
  subset every 5 % of the budget (learning curve), and on full validation splits at
  the end. The test split is never touched here (code/evaluate_test.py, later).
* The evaluated model is the final one (section 6); no best-checkpoint selection.

Run on Modal through modal_app.py, or locally for tests:
    python code/train.py --config configs/x.yaml --seed 0 --data_dir D --out_root O
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpb import evaluate_bpb  # noqa: E402
from config import Config, load_config  # noqa: E402,F401
from model import GPT, GPTConfig  # noqa: E402

# Peak dense bf16 tensor throughput, used only for the MFU diagnostic.
PEAK_FLOPS = {"L4": 121e12, "A100": 312e12, "H100": 989e12}


# ---------------------------------------------------------------------------
class BudgetClock:
    """Counted training time. Synchronises the GPU before every reading so queued
    asynchronous work is included."""

    def __init__(self, counted: float, device: str):
        self.counted, self.device, self._t0 = counted, device, None

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def start(self):
        self._sync()
        self._t0 = time.perf_counter()

    def pause(self):
        self._sync()
        self.counted += time.perf_counter() - self._t0
        self._t0 = None

    def now(self) -> float:
        if self._t0 is None:
            return self.counted
        self._sync()
        return self.counted + time.perf_counter() - self._t0


class WindowLoader:
    """Shuffled pass over non-overlapping windows of block_size+1 tokens.

    Window i covers tokens[i*T : i*T+T+1]; consecutive windows share one token so
    every training token is a target exactly once per pass."""

    def __init__(self, tokens: np.ndarray, T: int, seed: int):
        self.tokens, self.T, self.seed = tokens, T, seed
        self.n_windows = (len(tokens) - 1) // T
        if self.n_windows < 1:
            raise ValueError("not enough training tokens for one window")
        self.epoch, self.pos = 0, 0
        self._perm = self._make_perm(0)
        self._offsets = np.arange(T + 1)

    def _make_perm(self, epoch):
        return np.random.default_rng([self.seed, epoch]).permutation(self.n_windows)

    def next_batch(self, B: int) -> torch.Tensor:
        idx = []
        while len(idx) < B:
            take = min(B - len(idx), self.n_windows - self.pos)
            idx.extend(self._perm[self.pos:self.pos + take])
            self.pos += take
            if self.pos == self.n_windows:
                self.epoch += 1
                self.pos = 0
                self._perm = self._make_perm(self.epoch)
        starts = np.asarray(idx, dtype=np.int64) * self.T
        return torch.from_numpy(self.tokens[starts[:, None] + self._offsets].astype(np.int64))

    def passes(self) -> float:
        return self.epoch + self.pos / self.n_windows

    def state(self):
        return dict(epoch=self.epoch, pos=self.pos)

    def load_state(self, s):
        self.epoch, self.pos = s["epoch"], s["pos"]
        self._perm = self._make_perm(self.epoch)


def lr_at(cfg: Config, step: int, counted: float, t_warm_end: float | None) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "steps":
        span = max(1, cfg.max_steps - cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / span
    else:
        t0 = t_warm_end if t_warm_end is not None else counted
        progress = (counted - t0) / max(1e-9, cfg.budget_s - t0)
    progress = min(1.0, max(0.0, progress))
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def rng_state():
    s = dict(python=random.getstate(), numpy=np.random.get_state(),
             torch=torch.get_rng_state())
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
def run(cfg: Config, seed: int, data_dir: str, out_root: str, meta: dict | None = None,
        commit_fn=None) -> dict:
    """Train one run. `commit_fn` persists the output directory (Modal Volume commit)."""
    meta = dict(meta or {})
    commit_fn = commit_fn or (lambda: None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir, run_dir = Path(data_dir), Path(out_root) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_latest.pt"
    wall_start = time.time()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    ctx = (torch.autocast(device_type=device, dtype=ptdtype)
           if device == "cuda" and ptdtype != torch.float32 else contextlib.nullcontext())

    # ---- data (loaded into RAM before the clock starts; excluded from budget) ----
    tmeta = json.loads((data_dir / "tokenizer" / "meta.json").read_text())
    if tmeta["vocab_size"] != cfg.vocab_size:
        raise ValueError(f"tokenizer vocab {tmeta['vocab_size']} != config {cfg.vocab_size}")
    eot_id = tmeta["eot_id"]
    token_bytes = np.load(data_dir / "tokenizer" / "token_bytes.npy")
    train_tokens = np.fromfile(data_dir / "tokens" / cfg.corpus / "train.bin", dtype=np.uint16)
    train_tokens_available = len(train_tokens)
    if cfg.train_tokens_limit:
        if cfg.train_tokens_limit > len(train_tokens):
            raise ValueError("train_tokens_limit larger than the train split")
        train_tokens = train_tokens[:cfg.train_tokens_limit]
    val = {c: np.fromfile(data_dir / "tokens" / c / "val.bin", dtype=np.uint16)
           for c in sorted(set(cfg.eval_corpora) | {cfg.curve_corpus})}
    curve_tokens = val[cfg.curve_corpus][:cfg.curve_tokens + 1]

    # ---- model and optimiser (fresh, seeded) ----
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = GPT(GPTConfig(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                          block_size=cfg.block_size, vocab_size=cfg.vocab_size,
                          dropout=cfg.dropout, bias=cfg.bias)).to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2),
                                           device)
    loader = WindowLoader(train_tokens, cfg.block_size, seed)

    st = dict(step=0, counted=0.0, tokens=0, t_warm_end=None,
              next_eval=cfg.eval_every_frac * cfg.budget_s, next_ckpt=cfg.ckpt_every_s,
              curve=[], train_log=[], lr_log=[], step_times_first=[], attempts=[],
              eval_paused_s=0.0, ckpt_paused_s=0.0)
    if ckpt_path.exists():                       # resume: budget carries over
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["seed"] != seed or ck["config"] != dataclasses.asdict(cfg):
            raise RuntimeError(f"{ckpt_path} belongs to a different config/seed; refusing to resume")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        loader.load_state(ck["loader"])
        st = ck["state"]
        set_rng_state(ck["rng"])
        print(f"RESUMED from step {st['step']}, counted {st['counted']:.1f}s", flush=True)
    st["attempts"].append(dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               resumed_from_counted_s=st["counted"], resumed_from_step=st["step"]))

    raw_model = model
    if cfg.compile:
        model = torch.compile(model)

    def save_ckpt():
        atomic_torch_save(dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                               loader=loader.state(), state=st, rng=rng_state(), seed=seed,
                               config=dataclasses.asdict(cfg)), ckpt_path)
        commit_fn()

    def curve_point():
        r = evaluate_bpb(raw_model, curve_tokens, token_bytes, cfg.block_size, cfg.eval_batch,
                         eot_id, device, ctx)
        st["curve"].append(dict(step=st["step"], counted_s=round(st["counted"], 2),
                                tokens=st["tokens"], val_bpb=r["bpb"]))
        print(f"  [curve] step {st['step']} t={st['counted']:.0f}s val_bpb={r['bpb']:.4f}",
              flush=True)

    counts = raw_model.param_counts()
    tokens_per_step = cfg.micro_batch * cfg.block_size * cfg.grad_accum
    print(f"run {cfg.run_id} seed {seed} | params {counts} | tokens/step {tokens_per_step} | "
          f"train tokens {len(train_tokens):,} | device {device}", flush=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- training loop ----
    clock = BudgetClock(st["counted"], device)
    if st["step"] == 0 and not st["curve"]:
        curve_point()                           # step-0 point (clock not yet running)
    model.train()
    clock.start()
    last_log_t, loss_acc, loss_n = clock.now(), 0.0, 0
    crash_at = int(os.environ.get("CRASH_AT_STEP", "0"))   # tests only
    while True:
        t = clock.now()
        if t >= cfg.budget_s or (cfg.max_steps and st["step"] >= cfg.max_steps):
            break
        lr = lr_at(cfg, st["step"], t, st["t_warm_end"])
        for g in optimizer.param_groups:
            g["lr"] = lr
        t_step = time.perf_counter()
        for _ in range(cfg.grad_accum):
            batch = loader.next_batch(cfg.micro_batch)
            if device == "cuda":
                batch = batch.pin_memory().to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:].contiguous()   # nanoGPT's loss uses .view
            with ctx:
                _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            loss_acc += loss.item(); loss_n += 1
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        st["step"] += 1
        st["tokens"] += tokens_per_step
        now = clock.now()
        if len(st["step_times_first"]) < 60:            # compile warm-up diagnostic
            st["step_times_first"].append(round(time.perf_counter() - t_step, 4))
        if st["step"] == cfg.warmup_steps and st["t_warm_end"] is None:
            st["t_warm_end"] = now
        if crash_at and st["step"] == crash_at:
            raise SystemExit(f"CRASH_AT_STEP={crash_at} (test)")
        if now - last_log_t >= cfg.log_every_s or st["step"] == 1:
            st["train_log"].append(dict(step=st["step"], counted_s=round(now, 2),
                                        tokens=st["tokens"], lr=lr,
                                        train_loss=loss_acc / max(1, loss_n),
                                        passes=round(loader.passes(), 4)))
            print(f"step {st['step']} t={now:.0f}s lr={lr:.2e} "
                  f"loss={loss_acc / max(1, loss_n):.4f} tokens={st['tokens']:,}", flush=True)
            last_log_t, loss_acc, loss_n = now, 0.0, 0
        if st["step"] % 50 == 0:
            st["lr_log"].append((st["step"], round(now, 3), lr))
        do_eval = now >= st["next_eval"] and now < cfg.budget_s
        do_ckpt = (cfg.ckpt_every_steps and st["step"] % cfg.ckpt_every_steps == 0) or \
                  (not cfg.ckpt_every_steps and now >= st["next_ckpt"] and now < cfg.budget_s)
        if do_eval or do_ckpt:
            clock.pause()                                 # validation/checkpoint not counted
            st["counted"] = clock.counted
            p0 = time.perf_counter()
            if do_eval:
                model.eval(); curve_point(); model.train()
                while st["next_eval"] <= st["counted"]:
                    st["next_eval"] += cfg.eval_every_frac * cfg.budget_s
                st["eval_paused_s"] += time.perf_counter() - p0
            if do_ckpt:
                p1 = time.perf_counter()
                while st["next_ckpt"] <= st["counted"]:
                    st["next_ckpt"] += cfg.ckpt_every_s
                save_ckpt()
                st["ckpt_paused_s"] += time.perf_counter() - p1
            clock.start()
    clock.pause()
    st["counted"] = clock.counted

    # ---- final evaluation on full validation splits (outside the budget) ----
    model.eval()
    curve_point()
    final_val = {c: evaluate_bpb(raw_model, val[c], token_bytes, cfg.block_size, cfg.eval_batch,
                                 eot_id, device, ctx) for c in cfg.eval_corpora}
    for c, r in final_val.items():
        print(f"FINAL val {c}: bpb={r['bpb']:.4f} loss/token={r['loss_per_token']:.4f}", flush=True)
    torch.save(dict(model=raw_model.state_dict(), config=dataclasses.asdict(cfg), seed=seed),
               run_dir / "final_model.pt")
    save_ckpt()

    # ---- throughput: steady state between 10 % and 90 % of the run ----
    log = st["train_log"]
    mid = [r for r in log if 0.1 * st["counted"] <= r["counted_s"] <= 0.9 * st["counted"]]
    steady_tps = ((mid[-1]["tokens"] - mid[0]["tokens"]) / (mid[-1]["counted_s"] - mid[0]["counted_s"])
                  if len(mid) >= 2 and mid[-1]["counted_s"] > mid[0]["counted_s"] else None)
    gpu = meta.get("gpu_name") or (torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
    peak = next((v for k, v in PEAK_FLOPS.items() if k in gpu), None)
    result = dict(
        run_id=cfg.run_id, experiment=cfg.experiment, seed=seed, config=dataclasses.asdict(cfg),
        meta=meta, gpu_name=gpu, device=device,
        versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                      numpy=str(np.__version__), cuda=str(torch.version.cuda)),
        params=counts, flops_per_token=raw_model.flops_per_token(),
        tokens_per_step=tokens_per_step, steps=st["step"],
        counted_train_s=round(st["counted"], 2), budget_s=cfg.budget_s,
        wallclock_s_this_attempt=round(time.time() - wall_start, 1),
        attempts=st["attempts"], eval_paused_s=round(st["eval_paused_s"], 1),
        ckpt_paused_s=round(st["ckpt_paused_s"], 1),
        tokens_processed=st["tokens"], unique_train_tokens=int(len(train_tokens)),
        train_tokens_available=int(train_tokens_available),
        passes_over_data=round(st["tokens"] / (loader.n_windows * cfg.block_size), 4),
        avg_tokens_per_s=round(st["tokens"] / max(1e-9, st["counted"]), 1),
        steady_tokens_per_s=round(steady_tps, 1) if steady_tps else None,
        mfu_steady=(round(raw_model.estimate_mfu(steady_tps, peak), 4)
                    if steady_tps and peak else None),
        est_train_flops=raw_model.flops_per_token() * st["tokens"],
        first_step_s=st["step_times_first"][0] if st["step_times_first"] else None,
        step_times_first=st["step_times_first"],
        peak_memory_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 3)
                        if device == "cuda" else None),
        final_val={c: r for c, r in final_val.items()},
        final_val_bpb=final_val[cfg.eval_corpora[0]]["bpb"],
        best_curve_bpb_diagnostic=min(p["val_bpb"] for p in st["curve"]),
        curve=st["curve"], train_log=log, lr_log=st["lr_log"],
    )
    # Plain JSON types only: the result is sent back to a computer without torch installed
    # (torch.__version__ is a torch object; bug found in E00 benchmarks, 2026-11-01).
    result = json.loads(json.dumps(result, default=str))
    write_json(run_dir / "result.json", result)
    commit_fn()
    print(f"DONE {cfg.run_id}: {st['step']} steps, {st['tokens']:,} tokens, "
          f"{st['counted']:.1f}s counted, final val bpb {result['final_val_bpb']:.4f}", flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--set", nargs="*", default=[],
                   help="config overrides key=value (YAML-parsed), e.g. budget_s=120 compile=false")
    a = p.parse_args(argv)
    import yaml
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    cfg = load_config(a.config, overrides)
    run(cfg, a.seed, a.data_dir, a.out_root)


if __name__ == "__main__":
    main()
