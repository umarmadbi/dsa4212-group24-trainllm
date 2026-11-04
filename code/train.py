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

Added before Phase 1 (decision D07, 2026-11-01):
* Divergence guard: if the training loss becomes non-finite (NaN/inf) the run stops at
  once, is marked `diverged`, and gets no BPB (ranked worst). A diverged run would
  otherwise burn its whole budget producing NaN; LR pilots at the top of a grid can do this.
* Gradient norm (before clipping) is logged per interval, with the fraction of clipped
  steps: shows how close a learning rate is to instability, not only whether it diverged.
* GPU telemetry (SM clock, temperature, power, throttle reasons) is sampled every 30 s
  by a background thread calling nvidia-smi (CPU only, not on the GPU's critical path),
  to explain throughput differences between machines or over time (L4 power-throttling).

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
import subprocess
import sys
import threading
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


SUBSET_KEY = 1234   # fixed key for the nested random training subsets (E04, D23); never per-run


def subset_window_ids(n_tokens: int, T: int, subset_tokens: int) -> np.ndarray:
    """Window ids of the nested random subset with ~subset_tokens training targets.
    The permutation depends only on (SUBSET_KEY, number of windows), so for one train split
    every smaller subset is a prefix of every larger one (nested) and is a uniform random
    sample of the whole split. Sorted, so reading order is decided by the run's own loader."""
    n_win = (n_tokens - 1) // T
    m = subset_tokens // T
    if not 1 <= m <= n_win:
        raise ValueError(f"train_subset_tokens={subset_tokens} gives {m} windows; "
                         f"the split has {n_win}")
    return np.sort(np.random.default_rng(SUBSET_KEY).permutation(n_win)[:m])


class WindowLoader:
    """Shuffled pass over non-overlapping windows of block_size+1 tokens.

    Window i covers tokens[i*T : i*T+T+1]; consecutive windows share one token so
    every training token is a target exactly once per pass. If `window_ids` is given,
    only those windows are used (random training subset, E04)."""

    def __init__(self, tokens: np.ndarray, T: int, seed: int, window_ids=None):
        self.tokens, self.T, self.seed = tokens, T, seed
        self.window_ids = None if window_ids is None else np.asarray(window_ids, dtype=np.int64)
        self.n_windows = ((len(tokens) - 1) // T if self.window_ids is None
                          else len(self.window_ids))
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
        idx = np.asarray(idx, dtype=np.int64)
        if self.window_ids is not None:
            idx = self.window_ids[idx]
        starts = idx * self.T
        return torch.from_numpy(self.tokens[starts[:, None] + self._offsets].astype(np.int64))

    def passes(self) -> float:
        return self.epoch + self.pos / self.n_windows

    def state(self):
        return dict(epoch=self.epoch, pos=self.pos)

    def load_state(self, s):
        self.epoch, self.pos = s["epoch"], s["pos"]
        self._perm = self._make_perm(self.epoch)


class ShiftedSubsetLoader:
    """E04 follow-up F3 (D26): the same random subset as WindowLoader(window_ids=...), but the window
    BOUNDARIES move every pass, so the model does not see the identical (context, target) samples
    again (standard pipelines re-cut the text each epoch; ours repeated exact windows).

    The subset's windows (T tokens each, in file order) are joined into one stream with an EOT after
    each, marking the seam where two non-adjacent pieces of text meet (the same signal as a document
    boundary). Each pass draws a random offset o (from (seed, pass)) and cuts the circular stream into
    windows of T+1 tokens starting at o, o+T, o+2T, ...; window order is shuffled per pass. Every token
    of the stream is a target at most once per pass (all but < T per pass). The stream is ~1/T longer
    than the subset because of the EOT seams (0.2 % at T = 512)."""

    def __init__(self, tokens: np.ndarray, T: int, seed: int, window_ids, eot: int):
        ids = np.asarray(window_ids, dtype=np.int64)
        body = tokens[(ids * T)[:, None] + np.arange(T)]                       # (m, T)
        self.stream = np.concatenate([body, np.full((len(ids), 1), eot, dtype=tokens.dtype)],
                                     axis=1).reshape(-1)
        self.T, self.seed, self.L = T, seed, len(self.stream)
        self.n_windows = self.L // T
        self.epoch, self.pos = 0, 0
        self._set_epoch(0)

    def _set_epoch(self, epoch):
        rng = np.random.default_rng([self.seed, epoch, 7])
        self.offset = int(rng.integers(0, self.L))
        self._perm = rng.permutation(self.n_windows)

    def next_batch(self, B: int) -> torch.Tensor:
        idx = []
        while len(idx) < B:
            take = min(B - len(idx), self.n_windows - self.pos)
            idx.extend((self.offset + self._perm[self.pos:self.pos + take] * self.T).tolist())
            self.pos += take
            if self.pos == self.n_windows:
                self.epoch += 1
                self.pos = 0
                self._set_epoch(self.epoch)
        starts = np.asarray(idx, dtype=np.int64)
        pos = (starts[:, None] + np.arange(self.T + 1)) % self.L
        return torch.from_numpy(self.stream[pos].astype(np.int64))

    def passes(self) -> float:
        return self.epoch + self.pos / self.n_windows

    def state(self):
        return dict(epoch=self.epoch, pos=self.pos)

    def load_state(self, s):
        self.epoch, self.pos = s["epoch"], s["pos"]
        self._set_epoch(self.epoch)


def parse_mix(items) -> list[tuple[str, float]]:
    """["fineweb_edu=0.6", "dclm=0.4"] -> [("fineweb_edu", 0.6), ("dclm", 0.4)]; shares sum to 1."""
    out = []
    for it in items:
        name, _, share = str(it).partition("=")
        out.append((name.strip(), float(share)))
    if len(out) < 2 or any(w <= 0 for _, w in out) or abs(sum(w for _, w in out) - 1) > 1e-9:
        raise ValueError(f"corpus_mix needs >= 2 corpora with positive shares summing to 1: {items}")
    return out


def gather_windows(tokens: np.ndarray, T: int, ids: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Rows tokens[w*T : w*T+T+1] for every window id (2-D, read in chunks; works on a memmap)."""
    out = np.empty((len(ids), T + 1), dtype=np.uint16)
    off = np.arange(T + 1)
    for i in range(0, len(ids), chunk):
        out[i:i + chunk] = tokens[(ids[i:i + chunk] * T)[:, None] + off]
    return out


class MixLoader:
    """E03 follow-up R8 (D26): training batches mixed from several corpora in fixed token shares.

    Each corpus contributes a POOL: a uniform random set of its windows (the nested-subset
    permutation, key 1234) of `pool_tokens` tokens, loaded into RAM before the clock starts (the full
    splits of two corpora would not fit the 16 GB container). The pool is several times larger than
    what an hour reads, so nothing repeats, as in the single-corpus arms. Every window slot goes to the
    corpus furthest below its target share (deterministic, exact to within one window); within a
    corpus, windows are read in a shuffled pass seeded by (seed, corpus index, pass)."""

    def __init__(self, pools: list, shares: list, seed: int):
        self.pools, self.shares, self.seed = pools, np.asarray(shares, float), seed
        self.n = len(pools)
        self.drawn = np.zeros(self.n, dtype=np.int64)
        self.epoch = [0] * self.n
        self.pos = [0] * self.n
        self._perm = [self._make_perm(c, 0) for c in range(self.n)]
        self.n_windows = sum(len(p) for p in pools)

    def _make_perm(self, c, epoch):
        return np.random.default_rng([self.seed, 1000 + c, epoch]).permutation(len(self.pools[c]))

    def _next_row(self, c):
        row = self.pools[c][self._perm[c][self.pos[c]]]
        self.pos[c] += 1
        if self.pos[c] == len(self.pools[c]):
            self.epoch[c] += 1
            self.pos[c] = 0
            self._perm[c] = self._make_perm(c, self.epoch[c])
        return row

    def next_batch(self, B: int) -> torch.Tensor:
        rows = []
        for _ in range(B):
            total = self.drawn.sum() + 1
            c = int(np.argmax(self.shares * total - self.drawn))
            rows.append(self._next_row(c))
            self.drawn[c] += 1
        return torch.from_numpy(np.stack(rows).astype(np.int64))

    def passes(self) -> float:
        return float(max(self.epoch[c] + self.pos[c] / len(self.pools[c]) for c in range(self.n)))

    def state(self):
        return dict(drawn=self.drawn.tolist(), epoch=list(self.epoch), pos=list(self.pos))

    def load_state(self, s):
        self.drawn = np.asarray(s["drawn"], dtype=np.int64)
        self.epoch, self.pos = list(s["epoch"]), list(s["pos"])
        self._perm = [self._make_perm(c, self.epoch[c]) for c in range(self.n)]


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


class GpuTelemetry:
    """Samples nvidia-smi every `every_s` wall seconds in a daemon thread (CPU side only).
    Each sample: seconds since this attempt started, training step at that moment,
    SM clock (MHz), max SM clock, temperature (C), power draw / limit (W), GPU utilisation
    (%), and the active clock-event (throttle) reason bitmask."""

    QUERY = ("clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit,"
             "utilization.gpu,{reasons}")

    def __init__(self, samples: list, step_ref: dict, every_s: float = 30.0):
        self.samples, self.step_ref, self.every_s = samples, step_ref, every_s
        self._stop = threading.Event()
        self._t0 = time.time()
        self._reasons = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _query(self):
        for reasons in ([self._reasons] if self._reasons else
                        ["clocks_event_reasons.active", "clocks_throttle_reasons.active"]):
            out = subprocess.run(["nvidia-smi", f"--query-gpu={self.QUERY.format(reasons=reasons)}",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                self._reasons = reasons
                return [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
        return None

    def _loop(self):
        while not self._stop.wait(self.every_s):
            try:
                v = self._query()
            except Exception:  # noqa: BLE001  (telemetry must never break a run)
                v = None
            if not v:
                continue

            def num(s):
                try:
                    return float(s)
                except ValueError:
                    return None
            self.samples.append(dict(wall_s=round(time.time() - self._t0, 1),
                                     step=self.step_ref.get("step"),
                                     sm_mhz=num(v[0]), max_sm_mhz=num(v[1]), temp_c=num(v[2]),
                                     power_w=num(v[3]), power_limit_w=num(v[4]),
                                     util_pct=num(v[5]), reasons=v[6]))

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()


def telemetry_summary(samples: list) -> dict | None:
    """Median SM clock in the first and last 20 % of samples, max temperature, and how
    often the power cap was the active clock-limiting reason (bit 0x4 = SW power cap)."""
    s = [x for x in samples if x.get("sm_mhz")]
    if len(s) < 2:
        return None
    k = max(1, len(s) // 5)
    med = lambda xs: float(np.median(xs))  # noqa: E731
    def power_capped(x):
        try:
            return bool(int(str(x["reasons"]), 16) & 0x4)
        except ValueError:
            return False
    return dict(n_samples=len(s),
                sm_mhz_first20=med([x["sm_mhz"] for x in s[:k]]),
                sm_mhz_last20=med([x["sm_mhz"] for x in s[-k:]]),
                sm_mhz_max_rated=s[0].get("max_sm_mhz"),
                temp_c_max=max((x["temp_c"] or 0) for x in s),
                power_w_median=med([x["power_w"] for x in s if x["power_w"] is not None] or [0]),
                power_limit_w=s[0].get("power_limit_w"),
                frac_samples_power_capped=round(sum(power_capped(x) for x in s) / len(s), 3))


def _peak_rss_gb() -> dict | None:
    try:
        import resource
        return dict(self=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                    children_max=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                                       / 1e6, 2))
    except Exception:  # noqa: BLE001  (not available on Windows)
        return None


def rng_state():
    s = dict(python=random.getstate(), numpy=np.random.get_state(),
             torch=torch.get_rng_state())
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    # The checkpoint is loaded with map_location=<device>, which moves EVERY tensor in it to
    # the GPU, including these RNG states; PyTorch only accepts them as CPU ByteTensors.
    # (Bug found by the Modal kill-and-resume test, 2026-11-01: every real resume on a GPU
    # would have crashed. The local CPU test could not catch it. D08.)
    random.setstate(s["python"]); np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
def _default_crash():
    raise SystemExit("simulated preemption (test_crash_once_at_step)")


def run(cfg: Config, seed: int, data_dir: str, out_root: str, meta: dict | None = None,
        commit_fn=None, crash_fn=None) -> dict:
    """Train one run. `commit_fn` persists the output directory (Modal Volume commit).
    `crash_fn` is called for a simulated preemption (Modal passes a hard process kill)."""
    meta = dict(meta or {})
    commit_fn = commit_fn or (lambda: None)
    crash_fn = crash_fn or _default_crash
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
    mix = parse_mix(cfg.corpus_mix) if cfg.corpus_mix else None
    if mix:                                     # E03 R8 mixture (D26): pools, not full splits
        if cfg.corpus != "mix" or cfg.train_tokens_limit or cfg.train_subset_tokens:
            raise ValueError("corpus_mix needs corpus: mix and no train subset/limit")
        pools, mix_info = [], []
        for name, share in mix:
            mm = np.memmap(data_dir / "tokens" / name / "train.bin", dtype=np.uint16, mode="r")
            ids = subset_window_ids(len(mm), cfg.block_size, cfg.mix_pool_tokens)
            pools.append(gather_windows(mm, cfg.block_size, ids))
            mix_info.append(dict(corpus=name, share=share, pool_windows=int(len(ids)),
                                 split_tokens=int(len(mm)), first_window_ids=[int(x) for x in ids[:5]]))
            del mm
        train_tokens = np.zeros(1, dtype=np.uint16)          # unused in mix mode
        train_tokens_available = sum(m["split_tokens"] for m in mix_info)
    else:
        train_tokens = np.fromfile(data_dir / "tokens" / cfg.corpus / "train.bin", dtype=np.uint16)
        train_tokens_available = len(train_tokens)
    if cfg.train_tokens_limit:
        if cfg.train_tokens_limit > len(train_tokens):
            raise ValueError("train_tokens_limit larger than the train split")
        train_tokens = train_tokens[:cfg.train_tokens_limit]
    window_ids = None
    if cfg.train_subset_tokens:
        if cfg.train_tokens_limit:
            raise ValueError("use train_subset_tokens or train_tokens_limit, not both")
        window_ids = subset_window_ids(len(train_tokens), cfg.block_size, cfg.train_subset_tokens)
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
    if mix:
        loader = MixLoader(pools, [w for _, w in mix], seed)
        unique_tokens = loader.n_windows * cfg.block_size    # pooled windows (none repeats in 60 min)
    elif cfg.train_subset_shift:
        if window_ids is None:
            raise ValueError("train_subset_shift needs train_subset_tokens")
        loader = ShiftedSubsetLoader(train_tokens, cfg.block_size, seed, window_ids, eot_id)
        unique_tokens = loader.L                             # subset text + one EOT per seam
    else:
        loader = WindowLoader(train_tokens, cfg.block_size, seed, window_ids)
        # distinct training tokens available (unchanged meaning for full-split runs)
        unique_tokens = len(train_tokens) if window_ids is None else loader.n_windows * cfg.block_size

    st = dict(step=0, counted=0.0, tokens=0, t_warm_end=None,
              next_eval=cfg.eval_every_frac * cfg.budget_s, next_ckpt=cfg.ckpt_every_s,
              curve=[], train_log=[], lr_log=[], step_times_first=[], attempts=[],
              eval_paused_s=0.0, ckpt_paused_s=0.0, gpu_telemetry=[], diverged_at_step=None,
              n_clipped=0)
    if ckpt_path.exists():                       # resume: budget carries over
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["seed"] != seed or ck["config"] != dataclasses.asdict(cfg):
            raise RuntimeError(f"{ckpt_path} belongs to a different config/seed; refusing to resume")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        loader.load_state(ck["loader"])
        st = ck["state"]
        for k, v in (("gpu_telemetry", []), ("diverged_at_step", None), ("n_clipped", 0)):
            st.setdefault(k, v)                  # checkpoints from older code
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
          f"train tokens {unique_tokens:,} | device {device}", flush=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- training loop ----
    clock = BudgetClock(st["counted"], device)
    if st["step"] == 0 and not st["curve"]:
        curve_point()                           # step-0 point (clock not yet running)
    telemetry = None
    if device == "cuda":
        telemetry = GpuTelemetry(st["gpu_telemetry"], st)
        telemetry.start()
    model.train()
    clock.start()
    last_log_t, loss_acc, loss_n = clock.now(), 0.0, 0
    gn_acc, gn_max, gn_n, clip_n = 0.0, 0.0, 0, 0
    crash_at = int(os.environ.get("CRASH_AT_STEP", "0"))   # local tests only
    crash_once_at = cfg.test_crash_once_at_step if len(st["attempts"]) == 1 else 0
    while True:
        t = clock.now()
        if t >= cfg.budget_s or (cfg.max_steps and st["step"] >= cfg.max_steps):
            break
        if st["diverged_at_step"] is not None:
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
            loss_val = loss.item()
            loss_acc += loss_val; loss_n += 1
        # Gradient norm before clipping (computed with max_norm=inf when clipping is off).
        gnorm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf")).item()
        if not (math.isfinite(loss_val) and math.isfinite(gnorm)):
            # Divergence guard (D07): stop now; the run is ranked worst and gets no BPB.
            st["diverged_at_step"] = st["step"]
            optimizer.zero_grad(set_to_none=True)
            print(f"DIVERGED at step {st['step']} (loss {loss_val}, grad norm {gnorm}); "
                  f"stopping run", flush=True)
            break
        gn_acc += gnorm; gn_max = max(gn_max, gnorm); gn_n += 1
        if cfg.grad_clip > 0 and gnorm > cfg.grad_clip:
            clip_n += 1; st["n_clipped"] += 1
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
        if crash_once_at and st["step"] == crash_once_at:
            print(f"SIMULATED PREEMPTION at step {st['step']} (attempt 1)", flush=True)
            crash_fn()
        if now - last_log_t >= cfg.log_every_s or st["step"] == 1:
            st["train_log"].append(dict(step=st["step"], counted_s=round(now, 2),
                                        tokens=st["tokens"], lr=lr,
                                        train_loss=loss_acc / max(1, loss_n),
                                        grad_norm_mean=round(gn_acc / max(1, gn_n), 4),
                                        grad_norm_max=round(gn_max, 4),
                                        frac_clipped=round(clip_n / max(1, gn_n), 4),
                                        passes=round(loader.passes(), 4)))
            print(f"step {st['step']} t={now:.0f}s lr={lr:.2e} "
                  f"loss={loss_acc / max(1, loss_n):.4f} gnorm={gn_acc / max(1, gn_n):.3f} "
                  f"tokens={st['tokens']:,}", flush=True)
            last_log_t, loss_acc, loss_n = now, 0.0, 0
            gn_acc, gn_max, gn_n, clip_n = 0.0, 0.0, 0, 0
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
    if telemetry:
        telemetry.stop()
    diverged = st["diverged_at_step"] is not None

    # ---- final evaluation on full validation splits (outside the budget) ----
    model.eval()
    if diverged:
        final_val = {c: dict(bpb=None, diverged=True) for c in cfg.eval_corpora}
    else:
        curve_point()
        final_val = {c: evaluate_bpb(raw_model, val[c], token_bytes, cfg.block_size,
                                     cfg.eval_batch, eot_id, device, ctx)
                     for c in cfg.eval_corpora}
        for c, r in final_val.items():
            print(f"FINAL val {c}: bpb={r['bpb']:.4f} loss/token={r['loss_per_token']:.4f}",
                  flush=True)
    torch.save(dict(model=raw_model.state_dict(), config=dataclasses.asdict(cfg), seed=seed),
               run_dir / "final_model.pt")
    save_ckpt()

    # ---- throughput: steady state between 10 % and 90 % of the run ----
    log = st["train_log"]
    mid = [r for r in log if 0.1 * st["counted"] <= r["counted_s"] <= 0.9 * st["counted"]]
    steady_tps = ((mid[-1]["tokens"] - mid[0]["tokens"]) / (mid[-1]["counted_s"] - mid[0]["counted_s"])
                  if len(mid) >= 2 and mid[-1]["counted_s"] > mid[0]["counted_s"] else None)

    def tps_between(a, b):   # tokens/s between fractions a and b of the counted time
        seg = [r for r in log if a * st["counted"] <= r["counted_s"] <= b * st["counted"]]
        if len(seg) < 2 or seg[-1]["counted_s"] <= seg[0]["counted_s"]:
            return None
        return round((seg[-1]["tokens"] - seg[0]["tokens"])
                     / (seg[-1]["counted_s"] - seg[0]["counted_s"]), 1)
    # Throttling check: speed early (10-35 %) vs late (65-90 %) in the same run.
    tps_early, tps_late = tps_between(0.10, 0.35), tps_between(0.65, 0.90)
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
        tokens_processed=st["tokens"], unique_train_tokens=int(unique_tokens),
        train_subset=(dict(key=SUBSET_KEY, windows=int(len(window_ids)),
                           first_window_ids=[int(x) for x in window_ids[:5]],
                           shifted_windows=bool(cfg.train_subset_shift))
                      if window_ids is not None else None),
        corpus_mix=(dict(corpora=mix_info, windows_drawn=loader.state()["drawn"],
                         realised_share=[round(d / max(1, sum(loader.state()["drawn"])), 5)
                                         for d in loader.state()["drawn"]],
                         passes_per_corpus=[round(loader.epoch[c] + loader.pos[c] / len(pools[c]), 5)
                                            for c in range(loader.n)],
                         pool_tokens=cfg.mix_pool_tokens) if mix else None),
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
        # Peak host RAM of this process and of its children (torch.compile workers), GB
        # (Linux ru_maxrss is in KB). Used to size the Modal memory request (D12).
        peak_host_rss_gb=_peak_rss_gb(),
        steady_tokens_per_s_early=tps_early, steady_tokens_per_s_late=tps_late,
        final_val={c: r for c, r in final_val.items()},
        final_val_bpb=final_val[cfg.eval_corpora[0]]["bpb"],
        diverged=diverged, diverged_at_step=st["diverged_at_step"],
        frac_steps_clipped=round(st["n_clipped"] / max(1, st["step"]), 4),
        best_curve_bpb_diagnostic=min((p["val_bpb"] for p in st["curve"]
                                       if p["val_bpb"] is not None and
                                       math.isfinite(p["val_bpb"])), default=None),
        gpu_telemetry_summary=telemetry_summary(st["gpu_telemetry"]),
        gpu_telemetry=st["gpu_telemetry"],
        curve=st["curve"], train_log=log, lr_log=st["lr_log"],
    )
    # Plain JSON types only: the result is sent back to a computer without torch installed
    # (torch.__version__ is a torch object; bug found in E00 benchmarks, 2026-11-01).
    result = json.loads(json.dumps(result, default=str))
    write_json(run_dir / "result.json", result)
    commit_fn()
    bpb_txt = "DIVERGED" if diverged else f"{result['final_val_bpb']:.4f}"
    print(f"DONE {cfg.run_id}: {st['step']} steps, {st['tokens']:,} tokens, "
          f"{st['counted']:.1f}s counted, final val bpb {bpb_txt}", flush=True)
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
