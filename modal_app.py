"""
Modal launch file for the DSA4212 small-LLM project.

The GPU is chosen here, in code (gpu="L4"), not on the Modal website.
Matches DECLARATIONS.md section 1: 1 x NVIDIA L4, bf16, standard (preemptible) pricing.

Usage (from the repo root, after `pip install modal` and `modal setup`):
    modal run modal_app.py::gpu_check                     # ~1 min, confirms you got an L4
    modal run modal_app.py::prepare_data --train-mb 400   # data (CPU only)
    modal run modal_app.py::launch --configs configs/E00/bench_S3.yaml --seeds 0
    modal run modal_app.py::launch --configs "configs/E00/a.yaml,configs/E00/b.yaml"

`launch` runs each (config, seed) pair as its own job on its own L4, in parallel.
When a job finishes it writes results/runs/<run_id>__s<seed>.json on this computer
and appends a row to results/compute_ledger.csv, so every GPU run is recorded.
"""
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Container image: the software every run uses. Versions are pinned in
# requirements.txt, as DECLARATIONS.md section 1 requires.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")  # numpy 2.5 needs >=3.12
    .apt_install("gcc", "g++")  # torch.compile (Inductor/Triton) builds kernels at runtime
    # Pin Triton's floating-point fusion setting (its default changes in Triton 3.10),
    # so every run compiles the same way.
    .env({"TRITON_DEFAULT_FP_FUSION": "1"})
    .pip_install_from_requirements("requirements.txt")  # exact pinned versions
    .add_local_dir("code", remote_path="/root/code")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Persistent storage for the prepared data and run outputs.
# Each group member's account gets its own copy (same name, same contents).
data_vol = modal.Volume.from_name("small-llm-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("small-llm-runs", create_if_missing=True)

app = modal.App("dsa4212-small-llm", image=image)

GPU = "L4"  # <-- the declared hardware. Do not change for final runs.
L4_USD_PER_HOUR = 0.80  # for the ledger's cost estimate only


@app.function(gpu=GPU, timeout=10 * 60)
def gpu_check():
    """Quick check: prints the GPU name and confirms bf16 works. Costs ~1 cent."""
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    import torch
    name = torch.cuda.get_device_name(0)
    assert "L4" in name, f"Expected an L4, got {name}"
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    print(f"OK: {name}, bf16 supported = {torch.cuda.is_bf16_supported()}")


@app.function(
    gpu=GPU,
    cpu=4,
    memory=32768,  # the train split is loaded into RAM before the clock starts
    # A run needs 60 min of counted training plus paused validation/checkpointing,
    # compile and final evaluation, so allow 2 hours.
    timeout=2 * 60 * 60,
    # If the container is preempted or crashes, Modal restarts it; train.py resumes
    # from the last checkpoint with its counted time carried over.
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
    volumes={"/data": data_vol, "/runs": runs_vol},
)
def train(config: str, seed: int, meta: dict) -> dict:
    import sys
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert "L4" in gpu_name, f"Wrong GPU: {gpu_name}"  # DECLARATIONS.md section 1
    print(f"GPU: {gpu_name}")
    import traceback
    sys.path.insert(0, "/root/code")
    from train import load_config, run
    import hashlib
    import os
    cfg = load_config(f"/root/{config}", {})
    cfg.run_id = f"{cfg.run_id}__s{seed}"  # one output directory per (config, seed)
    meta = dict(meta, gpu_name=gpu_name, config_file=config,
                config_sha256=hashlib.sha256(Path(f"/root/{config}").read_bytes()).hexdigest())

    def hard_kill():
        # Simulated preemption: kill the process with no Python cleanup and no extra Volume
        # commit, like a lost machine. Only checkpoints committed earlier survive.
        os._exit(1)
    try:
        return run(cfg, seed, "/data", "/runs", meta=meta, commit_fn=runs_vol.commit,
                   crash_fn=hard_kill)
    except Exception:  # noqa: BLE001
        # A Python exception is a bug, not lost hardware: report it instead of raising,
        # so Modal does not retry it (retries are for preemption / container loss only).
        tb = traceback.format_exc()
        print(tb)
        return {"error": tb.strip().splitlines()[-1], "traceback": tb, "run_id": cfg.run_id}


@app.function(timeout=5 * 60, volumes={"/runs": runs_vol})
def archive_run_dir(run_id: str) -> str | None:
    """For a deliberate fresh rerun (`launch --fresh`): move an existing output directory
    to /runs/_replaced/<run_id>__<UTC time> so the new run cannot resume from it.
    Nothing is deleted. Returns the archive path, or None if there was nothing to move."""
    import shutil
    import time
    src = Path("/runs") / run_id
    if not src.exists():
        return None
    dst = Path("/runs/_replaced") / f"{run_id}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    runs_vol.commit()
    return str(dst)


@app.function(cpu=8, memory=32768, timeout=6 * 60 * 60,
              volumes={"/data": data_vol})
def prepare_data(stage: str = "all", train_mb: float = 400, eval_mb: float = 10,
                 tok_sample_mb: float = 200):
    """CPU-only (cheap): download, split, train tokenizer, tokenize. Phase 0.

    Phase 0 sample:  modal run modal_app.py::prepare_data --train-mb 400
    The full download (size decided from the L4 benchmarks) reruns with a larger
    --train-mb; validation/test splits are unchanged and train only grows.
    """
    subprocess.run(
        ["python", "/root/code/prepare_data.py", "--out=/data", f"--stage={stage}",
         f"--train_mb={train_mb}", f"--eval_mb={eval_mb}",
         f"--tok_sample_mb={tok_sample_mb}"],
        check=True,
    )
    data_vol.commit()


# ---------------------------------------------------------------------------
# Local side: runs on your computer, launches jobs and records their results.
# ---------------------------------------------------------------------------
# Everything that is shipped into the container and can change a result.
CODE_PATHS = ["code", "configs", "modal_app.py", "requirements.txt"]


def _git_info() -> dict:
    """Commit hash, whether the shipped code differs from that commit, and a SHA-256 of
    the shipped code itself.

    `dirty` includes UNTRACKED files under code/ and configs/ (D07): before this fix,
    new files that were never committed (e.g. train.py itself) did not count as dirty,
    so a run could be logged as 'clean' while using code that is in no commit."""
    import hashlib

    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS))
    h = hashlib.sha256()
    files = []
    for p in CODE_PATHS:
        pp = Path(p)
        files += [pp] if pp.is_file() else sorted(f for f in pp.rglob("*") if f.is_file()
                                                   and "__pycache__" not in f.parts)
    for f in sorted(files, key=lambda f: f.as_posix()):
        h.update(f.as_posix().encode()); h.update(b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n")); h.update(b"\0")
    return dict(git_commit=commit, git_dirty=dirty, code_sha256=h.hexdigest())


def _modal_workspace() -> str:
    out = subprocess.run(["modal", "profile", "current"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def _bpb_txt(result: dict) -> str:
    if result.get("diverged"):
        return f"DIVERGED at step {result.get('diverged_at_step')}"
    return f"{result['final_val_bpb']:.4f}"


LEDGER_FIELDS = ["run_id", "experiment", "kind", "date_sgt", "member", "modal_workspace",
                 "gpu_name", "git_commit", "config", "seed", "counted_train_s", "wallclock_s",
                 "tokens_processed", "status", "cost_usd", "notes"]


def _record(result: dict | None, config: str, seed: int, meta: dict, kind: str,
            started: dt.datetime, error: str | None = None):
    sgt = dt.timezone(dt.timedelta(hours=8))
    # Container time of the (last) attempt; falls back to time since launch for failures
    # (which then includes queueing, so it overestimates).
    wall = (result["wallclock_s_this_attempt"] if result
            else (dt.datetime.now(sgt) - started).total_seconds())
    commit = meta["git_commit"][:7] + ("+dirty" if meta["git_dirty"] else "")
    if result is not None:
        out = Path("results/runs") / f"{result['run_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            # A rerun (--fresh replacement) has the same run_id: keep the old result
            # (docs/README rule 3: never delete or overwrite a result).
            old = Path("results/runs/_replaced") / (
                f"{result['run_id']}__{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.json")
            old.parent.mkdir(parents=True, exist_ok=True)
            out.rename(old)
            print(f"kept previous result as {old}")
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"saved {out}")
    row = dict(
        run_id=result["run_id"] if result else f"{Path(config).stem}__s{seed}",
        experiment=result["experiment"] if result else "",
        kind=kind, date_sgt=started.strftime("%Y-%m-%d %H:%M"), member=meta["member"],
        modal_workspace=meta["modal_workspace"],
        gpu_name=result["gpu_name"] if result else "", git_commit=commit, config=config,
        seed=seed, counted_train_s=result["counted_train_s"] if result else "",
        wallclock_s=round(wall), tokens_processed=result["tokens_processed"] if result else "",
        status=("diverged" if result and result.get("diverged")
                else "completed" if result else "failed"),
        cost_usd=f"~{wall / 3600 * L4_USD_PER_HOUR:.2f}",
        notes=((f"final val bpb {_bpb_txt(result)}; attempts {len(result['attempts'])}"
                + (f"; fresh rerun, old dir archived to {meta['archived_to']}"
                   if meta.get("archived_to") else "")
                + (f"; code sha {meta['code_sha256'][:10]}" if meta.get("code_sha256") else ""))
               if result else f"ERROR: {error}"[:300]),
    )
    ledger = Path("results/compute_ledger.csv")
    with open(ledger, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=LEDGER_FIELDS).writerow(row)


@app.local_entrypoint()
def launch(configs: str, seeds: str = "0", kind: str = "pilot", member: str = "umarmadbi",
           fresh: bool = False, allow_dirty: bool = False):
    """Run every (config, seed) pair in parallel, each on its own L4.

    A (config, seed) whose output directory already exists on the runs Volume RESUMES from
    its checkpoint (right after a crash). For a deliberate rerun (e.g. replacing a run on a
    slow machine) pass --fresh: the old directory is archived first, never deleted.
    Refuses to launch from code that differs from the last commit unless --allow-dirty
    (DECLARATIONS section 1: every run launches from a committed version; D06/D07)."""
    import glob as _glob
    cfg_list = []
    for c in (c.strip().replace("\\", "/") for c in configs.split(",") if c.strip()):
        matches = sorted(_glob.glob(c)) if any(ch in c for ch in "*?[") else [c]
        if not matches:
            raise FileNotFoundError(f"no config matches {c}")
        cfg_list.extend(matches)
    cfg_list = [c.replace("\\", "/") for c in cfg_list]
    seed_list = [int(s) for s in seeds.split(",")]
    # Validate every config HERE, before any GPU starts (catches typos and wrong types).
    import sys
    sys.path.insert(0, str(Path("code").resolve()))
    from config import load_config
    for c in cfg_list:
        if not Path(c).exists():
            raise FileNotFoundError(c)
        load_config(c)
    print(f"{len(cfg_list)} config(s) validated locally")
    meta = dict(_git_info(), member=member, modal_workspace=_modal_workspace(), kind=kind)
    if meta["git_dirty"]:
        if not allow_dirty:
            raise SystemExit("REFUSING TO LAUNCH: code/, configs/, modal_app.py or "
                             "requirements.txt differ from the last commit (or have untracked "
                             "files). Commit first (`git status` shows what), or pass "
                             "--allow-dirty for a throwaway test; it is then marked +dirty.")
        print("WARNING: launching uncommitted code (--allow-dirty); the ledger marks it +dirty.")
    sgt = dt.timezone(dt.timedelta(hours=8))
    started = dt.datetime.now(sgt)
    jobs = [(c, s) for c in cfg_list for s in seed_list]
    job_meta = {}
    for c, s in jobs:
        jm = dict(meta)
        if fresh:
            rid = f"{load_config(c).run_id}__s{s}"
            archived = archive_run_dir.remote(rid)
            if archived:
                print(f"--fresh: archived old {rid} -> {archived}")
                jm["archived_to"] = archived
        job_meta[(c, s)] = jm
    calls = [(c, s, train.spawn(c, s, job_meta[(c, s)])) for c, s in jobs]
    print(f"launched {len(calls)} job(s): " + ", ".join(f"{c}#s{s}" for c, s, _ in calls))
    for c, s, call in calls:
        try:
            res = call.get()
            if "error" in res:
                Path("results/failed").mkdir(parents=True, exist_ok=True)
                Path(f"results/failed/{res['run_id']}.txt").write_text(res["traceback"])
                raise RuntimeError(res["error"])
            _record(res, c, s, job_meta[(c, s)], kind, started)
            print(f"done {res['run_id']}: final val bpb {_bpb_txt(res)}, "
                  f"{res['steady_tokens_per_s']} tokens/s steady")
        except Exception as e:  # noqa: BLE001  (record failed runs too)
            _record(None, c, s, job_meta[(c, s)], kind, started, error=repr(e))
            print(f"FAILED {c} seed {s}: {e!r}")


@app.local_entrypoint()
def recover(run_ids: str, configs: str, seeds: str = "0", kind: str = "benchmark",
            member: str = "umarmadbi"):
    """Fetch result.json of finished runs from the runs Volume and record them locally.

    For runs that trained successfully but whose result did not reach this computer.
    run_ids / configs are comma-separated, in the same order.
        modal run modal_app.py::recover --run-ids "E00-bench_S1__s0" --configs "configs/E00/bench_S1.yaml"
    """
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    cfgs = [c.strip().replace("\\", "/") for c in configs.split(",") if c.strip()]
    seed_list = [int(s) for s in seeds.split(",")]
    if len(seed_list) == 1:
        seed_list = seed_list * len(ids)
    assert len(ids) == len(cfgs) == len(seed_list), "run_ids, configs, seeds must align"
    meta = dict(_git_info(), member=member, modal_workspace=_modal_workspace(), kind=kind)
    sgt = dt.timezone(dt.timedelta(hours=8))
    for rid, c, s in zip(ids, cfgs, seed_list):
        try:
            raw = b"".join(runs_vol.read_file(f"{rid}/result.json"))
        except Exception as e:  # noqa: BLE001
            print(f"no result for {rid}: {e!r}")
            continue
        res = json.loads(raw)
        # commit/meta recorded inside the result are the ones the run actually used
        run_meta = dict(meta, git_commit=res["meta"].get("git_commit", meta["git_commit"]),
                        git_dirty=res["meta"].get("git_dirty", meta["git_dirty"]))
        started = dt.datetime.fromisoformat(res["attempts"][0]["started_utc"].replace("Z", "+00:00")
                                            ).astimezone(sgt)
        _record(res, c, s, run_meta, kind, started)
        print(f"recovered {rid}: final val bpb {_bpb_txt(res)}, "
              f"{res['steady_tokens_per_s']} tokens/s steady")
