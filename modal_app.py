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

_INPUTS_IN_THIS_CONTAINER = 0  # counts runs served by one container process (should stay 1)

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
    # Host RAM: the train split (~4.9 GB) is loaded before the clock starts. Measured peak of a
    # 60-min N26 run: 9.6 GB (E01 horizon runs, D12 logging). 16 GB leaves ~65 % headroom and
    # may fit on more L4 machines than the 32 GB used until E01 (D14). Training is unaffected.
    memory=16384,
    # A run needs 60 min of counted training plus paused validation/checkpointing,
    # compile and final evaluation, so allow 2 hours.
    timeout=2 * 60 * 60,
    # If the container is preempted or crashes, Modal restarts it; train.py resumes
    # from the last checkpoint with its counted time carried over.
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
    volumes={"/data": data_vol, "/runs": runs_vol},
    # One run per container (D08). Modal otherwise reuses a finished container for the next
    # queued run, which then finds torch.compile's kernels already cached and skips the
    # ~20-45 s compile warm-up: it gets that time as extra training inside the same budget.
    # Found in E01 (N11 lr2e-3 and lr8e-3: first step 0.11 s instead of ~27 s).
    single_use_containers=True,
)
def train(config: str, seed: int, meta: dict) -> dict:
    import sys
    global _INPUTS_IN_THIS_CONTAINER
    _INPUTS_IN_THIS_CONTAINER += 1
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
                inputs_in_this_container=_INPUTS_IN_THIS_CONTAINER,  # must be 1 (D08)
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
                 tok_sample_mb: float = 200, pin_revisions: str = ""):
    """CPU-only (cheap): download, split, train tokenizer, tokenize. Phase 0.

    Phase 0 sample:  modal run modal_app.py::prepare_data --train-mb 400
    The full download (size decided from the L4 benchmarks) reruns with a larger
    --train-mb; validation/test splits are unchanged and train only grows.
    """
    subprocess.run(
        ["python", "/root/code/prepare_data.py", "--out=/data", f"--stage={stage}",
         f"--train_mb={train_mb}", f"--eval_mb={eval_mb}",
         f"--tok_sample_mb={tok_sample_mb}", f"--pin_revisions={pin_revisions}"],
        check=True,
    )
    data_vol.commit()


@app.function(cpu=4, memory=8192, timeout=60 * 60, volumes={"/data": data_vol})
def hash_data_files(paths: list[str]) -> dict:
    """SHA-256 of files on this workspace's data Volume (None if missing)."""
    import hashlib
    out = {}
    for p in paths:
        f = Path(p)
        if not f.is_file():
            out[p] = None
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        out[p] = h.hexdigest()
    return out


def _expected_hashes() -> dict:
    """Every file training reads, with the hash recorded in data/manifest.json."""
    m = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    exp = {"/data/tokenizer/tokenizer.json": m["tokenizer"]["tokenizer_json_sha256"],
           "/data/tokenizer/token_bytes.npy": m["tokenizer"]["token_bytes_sha256"]}
    for splits in m["tokens"]["splits"].values():
        for e in splits.values():
            exp[e["file"]] = e["sha256"]
    return exp


@app.local_entrypoint()
def verify_data():
    """DECLARATIONS 'Running across several Modal accounts': before any final run on a
    workspace, every token file and the tokenizer on its Volume must match data/manifest.json.
        modal run modal_app.py::verify_data"""
    exp = _expected_hashes()
    got = hash_data_files.remote(sorted(exp))
    bad = [p for p in exp if got.get(p) != exp[p]]
    for p in sorted(exp):
        status = "OK" if p not in bad else ("MISSING" if got.get(p) is None else "MISMATCH")
        print(f"{status:8s} {p}")
    print(f"workspace profile: {_modal_workspace()}")
    if bad:
        raise SystemExit(f"DATA CHECK FAILED: {len(bad)} of {len(exp)} files differ. "
                         "Do not run experiments on this workspace.")
    print(f"DATA CHECK PASSED: all {len(exp)} files match data/manifest.json")


@app.local_entrypoint()
def rebuild_data():
    """Build this workspace's copy of the data with exactly the settings AND dataset
    versions recorded in data/manifest.json (D10). For a new group member's workspace.
    CPU only, ~1 hour, ~$0.60. ALWAYS run it detached, then check the hashes later:
        modal run --detach modal_app.py::rebuild_data
        (about an hour later)  modal run modal_app.py::verify_data

    The build is handed off with .spawn() and this command returns at once (D16). The
    first version waited on it with .remote(); when the laptop lost its connection, Modal
    cancelled the build even with --detach (2026-11-03, joshuahanrc workspace). Spawned
    calls survived a dropped connection before (E01 horizon run), so the build no longer
    depends on the laptop staying online. Without --detach the app would stop as soon as
    this command returns and cancel the build, so --detach is required."""
    m = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    a = m["args"]
    pins = {c: e["revision"] for c, e in m["corpora"].items()}
    print(f"workspace profile: {_modal_workspace()}")
    print(f"rebuilding with train_mb={a['train_mb']} eval_mb={a['eval_mb']} "
          f"tok_sample_mb={a['tok_sample_mb']}, revisions pinned: {pins}")
    call = prepare_data.spawn(stage="all", train_mb=a["train_mb"], eval_mb=a["eval_mb"],
                              tok_sample_mb=a["tok_sample_mb"],
                              pin_revisions=json.dumps(pins))
    print(f"build started on Modal (call {call.object_id}). It runs ~1 hour whether or not "
          f"this computer stays online, PROVIDED this was launched with --detach.\n"
          f"Then check it with:  modal run modal_app.py::verify_data")


# ---------------------------------------------------------------------------
# Analysis jobs (outside every training budget; D25). Results are written to the runs Volume
# FIRST (/runs/_analysis/...), so a dropped laptop connection loses nothing: fetch them later.
# ---------------------------------------------------------------------------
CONTAM_TRAIN = ["fineweb_edu", "fineweb", "dclm"]


@app.function(cpu=8, memory=32768, timeout=3 * 60 * 60,
              volumes={"/data": data_vol, "/runs": runs_vol})
def contamination_one(train_corpus: str) -> dict:
    """13-gram check of every val/test document against ONE training corpus's full train split."""
    import sys
    sys.path.insert(0, "/root/code")
    import contamination as C
    res = C.run_check("/data", train_corpora=(train_corpus,))
    out = Path("/runs/_analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"contamination_{train_corpus}.json").write_text(json.dumps(res))
    runs_vol.commit()
    return res


@app.local_entrypoint()
def contamination(fetch_only: bool = False):
    """E03 rule R3 / DECLARATIONS section 3 (D25). CPU only, ~10-20 min, a few cents.
        modal run --detach modal_app.py::contamination        # starts 3 jobs (one per train corpus)
        modal run modal_app.py::contamination --fetch-only    # later, if the connection dropped
    Writes results/contamination.json and prints the summary. Any verified workspace works (the data
    are identical); fetch from the SAME workspace."""
    import sys
    sys.path.insert(0, str(Path("code").resolve()))
    import contamination as C
    print(f"workspace profile: {_modal_workspace()}")
    if not fetch_only:
        calls = [(tc, contamination_one.spawn(tc)) for tc in CONTAM_TRAIN]
        print(f"started {len(calls)} job(s); if this terminal loses its connection, the jobs continue: "
              "fetch later with --fetch-only")
        for tc, call in calls:
            try:
                call.get()
                print(f"done: {tc}")
            except Exception as e:  # noqa: BLE001
                print(f"waiting for {tc} failed here ({e!r}); fetch later with --fetch-only")
    merged = None
    for tc in CONTAM_TRAIN:
        try:
            r = json.loads(b"".join(runs_vol.read_file(f"_analysis/contamination_{tc}.json")))
        except Exception as e:  # noqa: BLE001
            print(f"not available yet: {tc} ({e!r})")
            continue
        if merged is None:
            merged = {k: v for k, v in r.items() if k != "by_train_corpus"}
            merged["by_train_corpus"] = {}
        merged["by_train_corpus"].update(r["by_train_corpus"])
    if not merged:
        return
    merged["union_declared"] = C.union(merged)
    merged["complete"] = sorted(merged["by_train_corpus"]) == sorted(CONTAM_TRAIN)
    merged["modal_workspace"] = _modal_workspace()
    merged.update({k: v for k, v in _git_info().items() if k in ("git_commit", "git_dirty")})
    Path("results").mkdir(exist_ok=True)
    Path("results/contamination.json").write_text(json.dumps(merged, indent=1))
    print(C.report(merged))
    print("written: results/contamination.json" + ("" if merged["complete"] else "  (INCOMPLETE)"))


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=30 * 60,
              volumes={"/data": data_vol, "/runs": runs_vol})
def rescore_one(run_id: str, sets: list, exclude: dict, tag: str, stride: int = 0) -> dict:
    """Score /runs/<run_id>/final_model.pt on full validation splits, optionally without some docs."""
    import sys
    sys.path.insert(0, "/root/code")
    import rescore as R
    if not (Path("/runs") / run_id / "final_model.pt").exists():
        return {"error": f"{run_id}/final_model.pt not on this workspace's runs Volume"}
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    res = R.rescore(f"/runs/{run_id}", "/data", sets, exclude, stride=stride)
    res.update(gpu_name=gpu_name, tag=tag)
    out = Path("/runs/_analysis/rescore")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}__{tag}.json").write_text(json.dumps(res))
    runs_vol.commit()
    return res


@app.local_entrypoint()
def rescore(run_ids: str, sets: str = "fineweb_edu,fineweb,dclm,wikipedia", exclude: str = "",
            tag: str = "", fetch_only: bool = False, stride: int = 0):
    """Re-score saved final models (outside the budget; D25). Run from the workspace that trained them.
        modal run modal_app.py::rescore --run-ids "E03-N49_dclm__s0,E03-N49_dclm__s1" --exclude contamination
        modal run modal_app.py::rescore --run-ids "E01-N26_lr4e-3_60min__s0"
    --exclude contamination: drop, from each validation set, the documents listed in
    results/contamination.json (union over training corpora, declared threshold), the same for every run.
    Writes results/rescore/<run_id>__<tag>.json (tag: 'nocontam' or 'full' unless given).
    If the terminal loses its connection, rerun the same command with --fetch-only (same workspace).
    --stride N (E07 diagnostic, D36): also score with overlapping windows every N tokens (tag 'slide<N>').
        modal run modal_app.py::rescore --run-ids "E07-V4096_lr2e-3_60min__s0" --sets fineweb_edu --stride 128"""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    set_list = [x.strip() for x in sets.split(",") if x.strip()]
    ex = {}
    if exclude == "contamination":
        u = json.loads(Path("results/contamination.json").read_text())["union_declared"]
        ex = {name.split("/")[0]: v["doc_indices"] for name, v in u.items()
              if name.endswith("/val") and v["doc_indices"]}
        print("excluding: " + (", ".join(f"{c} {len(v)} docs" for c, v in ex.items()) or "nothing"))
    elif exclude:
        raise SystemExit("--exclude must be '' or 'contamination'")
    tag = tag or ("nocontam" if exclude else f"slide{stride}" if stride else "full")
    print(f"workspace profile: {_modal_workspace()}")
    Path("results/rescore").mkdir(parents=True, exist_ok=True)
    if fetch_only:      # D26: results are saved on the runs Volume first; fetch them after a dropped connection
        calls = [(rid, None) for rid in ids]
    else:
        calls = [(rid, rescore_one.spawn(rid, set_list, ex, tag, stride)) for rid in ids]
    for rid, call in calls:
        try:
            res = (json.loads(b"".join(runs_vol.read_file(f"_analysis/rescore/{rid}__{tag}.json")))
                   if call is None else call.get())
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {rid}: {e!r} (if the job finished, fetch it later with --fetch-only "
                  "from the same workspace)")
            continue
        if "error" in res:
            print(f"SKIPPED {rid}: {res['error']} (run it from the workspace that trained it)")
            continue
        Path(f"results/rescore/{rid}__{tag}.json").write_text(json.dumps(res, indent=1))
        for c, e in res["sets"].items():
            d = e.get("diff_vs_stored")
            print(f"{rid} {c:12s} bpb {e['full']['bpb']:.4f}"
                  + (f" (stored {e['stored_bpb']:.4f}, diff {d:+.1e})" if d is not None else " (not scored before)")
                  + (f"  without {e['excluded']['docs_removed']} docs: {e['excluded']['bpb']:.4f}"
                     if "excluded" in e else "")
                  + (f"  sliding (stride {e['sliding']['stride']}): {e['sliding']['bpb']:.4f}"
                     if "sliding" in e else ""))


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
        # A failed run's only timing is time since launch, which includes queueing (one E01 row
        # logged ~$6 for a job that never started, D12), so no cost is guessed for it.
        cost_usd=(f"~{wall / 3600 * L4_USD_PER_HOUR:.2f}" if result
                  else "not estimated (failed; see Modal usage page)"),
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
        # The run's own metadata (recorded at launch) wins over this terminal's: commit, dirty
        # flag, code hash, workspace and member. Before D18 only commit/dirty were taken from
        # the result, so a run recovered from a terminal on a different Modal profile (or after
        # new code was written) got the wrong workspace / code hash in its ledger row.
        run_meta = dict(meta, **{k: res["meta"][k] for k in
                                 ("git_commit", "git_dirty", "code_sha256", "modal_workspace",
                                  "member", "archived_to") if k in res.get("meta", {})})
        if run_meta["modal_workspace"] != meta["modal_workspace"]:
            print(f"note: {rid} ran on workspace {run_meta['modal_workspace']}, this terminal is "
                  f"{meta['modal_workspace']} (the run's own workspace is recorded)")
        started = dt.datetime.fromisoformat(res["attempts"][0]["started_utc"].replace("Z", "+00:00")
                                            ).astimezone(sgt)
        _record(res, c, s, run_meta, kind, started)
        print(f"recovered {rid}: final val bpb {_bpb_txt(res)}, "
              f"{res['steady_tokens_per_s']} tokens/s steady")
