"""
One-time TEST-SET scoring behind the test lock (DECLARATIONS section 5; D42).

Plain language: the test split was written once in Phase 0 and never read since. Every design decision
(learning rates, sizes, corpora, optimiser, shape, vocabulary, the final recipe) was made on VALIDATION
data. At the very end, the models whose results the report gives are scored on the test split exactly
once, so the reported numbers are not tuned to the data they are measured on. This file is the lock:

* it refuses any run that is not listed in `configs/frozen_configs.yaml`;
* it refuses while that file's `status` is not `frozen` (the file is drafted first, frozen after the last
  E08 decision, and committed BEFORE any test scoring: its git history is the evidence);
* the Modal entry point (`modal_app.py::test_eval`) also refuses if the frozen file or any code differs from
  the last commit, and refuses a run already scored (on the laptop AND on the workspace's runs Volume), so a
  test score can never be re-computed and "improved".

What is computed per model (all outside the training budget; the declared protocol of `bpb.evaluate_bpb`):
* test BPB on the sets its entry lists (FineWeb-Edu = the declared primary result; FineWeb, DCLM, Wikipedia
  give the four-set average AVG4, the declared metric of the data-quality comparison);
* the same with the contaminated test documents removed (declared 13-gram rule, the union over training
  corpora in results/contamination.json; the SAME documents for every model, as rule R3 did for validation),
  because three test sets are above the 1 % threshold;
* optionally overlapping-window BPB (stride 128) on FineWeb-Edu test: the D8 caveat for vocabulary results;
* optionally a few text samples from fixed prompts (illustration only, never a result).

A model trained on a vocabulary variant is scored on that variant's files (`rescore.data_dir_for`): the same
documents in the same order, tokenized by its own tokenizer.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

FROZEN = Path("configs/frozen_configs.yaml")
TIER_DEFAULTS = {
    # core: every comparison the report headlines (baseline, Phase 4-6 winners, final recipes)
    "core": dict(sets=["fineweb_edu", "fineweb", "dclm", "wikipedia"], sliding=128),
    # data: E03 corpora (same 8k tokenizer as the baseline, so no sliding view is needed)
    "data": dict(sets=["fineweb_edu", "fineweb", "dclm", "wikipedia"], sliding=0),
    # extra: other completed 60-min runs (E02 sizes, E04 repetition): primary set only
    "extra": dict(sets=["fineweb_edu"], sliding=0),
}
SAMPLE_PROMPTS = [          # declared with D42, before any sample was generated
    "The water cycle describes how water",
    "The main causes of the French Revolution were",
    "To solve a quadratic equation, you can",
]
SAMPLE_SETTINGS = dict(max_new_tokens=100, temperature=0.8, top_k=50, seed=1234)


class LockError(RuntimeError):
    """The test lock refused (never caught inside this module)."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_frozen(text: str) -> dict:
    """Parse the frozen list. Returns {'status', 'runs': {run_id: entry with tier defaults applied}}."""
    import yaml
    d = yaml.safe_load(text) or {}
    runs = {}
    for rid, e in (d.get("runs") or {}).items():
        e = dict(e or {})
        tier = e.get("tier")
        if tier not in TIER_DEFAULTS:
            raise ValueError(f"{rid}: tier must be one of {sorted(TIER_DEFAULTS)}, got {tier!r}")
        full = dict(TIER_DEFAULTS[tier]); full.update(e)
        if not full.get("workspace"):
            raise ValueError(f"{rid}: no workspace (the Modal workspace that holds its final model)")
        if "__s" not in rid:
            raise ValueError(f"{rid}: run ids must include the seed (<run_id>__s<seed>)")
        runs[rid] = full
    return dict(status=d.get("status"), runs=runs)


def check_lock(run_ids, frozen: dict) -> None:
    """The lock itself: status frozen, every requested run listed. Raises LockError otherwise."""
    if frozen.get("status") != "frozen":
        raise LockError(f"test lock: configs/frozen_configs.yaml has status {frozen.get('status')!r}, not 'frozen' "
                        "(freeze it after the last E08 decision, commit, then score)")
    missing = [r for r in run_ids if r not in frozen["runs"]]
    if missing:
        raise LockError(f"test lock: not in configs/frozen_configs.yaml: {missing}")


def test_exclusions(contamination: dict) -> dict:
    """{set: [doc indices]} of contaminated TEST documents (declared threshold, union over training corpora)."""
    u = contamination["union_declared"]
    return {name.split("/")[0]: v["doc_indices"] for name, v in u.items()
            if name.endswith("/test") and v["doc_indices"]}


def generate_samples(model, tokenizer_json: Path, device: str, ctx) -> list:
    """Illustration only: fixed prompts, temperature, top-k and seed (SAMPLE_*)."""
    import torch
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    out = []
    for i, p in enumerate(SAMPLE_PROMPTS):
        torch.manual_seed(SAMPLE_SETTINGS["seed"] + i)
        if device == "cuda":
            torch.cuda.manual_seed(SAMPLE_SETTINGS["seed"] + i)
        ids = torch.tensor([tok.encode(p).ids], dtype=torch.long, device=device)
        with torch.no_grad(), ctx:
            y = model.generate(ids, SAMPLE_SETTINGS["max_new_tokens"], temperature=SAMPLE_SETTINGS["temperature"],
                               top_k=SAMPLE_SETTINGS["top_k"])
        out.append(dict(prompt=p, text=tok.decode(y[0].tolist())))
    return out


def evaluate_test(run_id: str, runs_root: str, data_root: str, frozen_text: str, contamination: dict,
                  device: str | None = None) -> dict:
    """Score one frozen run on the test split. Refuses (LockError) unless the lock allows it."""
    import contextlib
    import torch
    import rescore as R
    frozen = load_frozen(frozen_text)
    check_lock([run_id], frozen)
    e = frozen["runs"][run_id]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(runs_root) / run_id
    excl = {s: v for s, v in test_exclusions(contamination).items() if s in e["sets"]}
    res = R.rescore(str(run_dir), data_root, e["sets"], excl, device=device, stride=0, split="test")
    if e.get("sliding"):
        s1 = R.rescore(str(run_dir), data_root, ["fineweb_edu"], {}, device=device, stride=int(e["sliding"]),
                       split="test")
        res["sets"]["fineweb_edu"]["sliding"] = s1["sets"]["fineweb_edu"]["sliding"]
    if e.get("samples"):
        model, c, _ = R.load_model(run_dir / "final_model.pt", device)
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device == "cuda" and c.get("dtype") == "bfloat16" else contextlib.nullcontext())
        res["samples"] = generate_samples(model, R.data_dir_for(c, data_root) / "tokenizer" / "tokenizer.json",
                                          device, ctx)
        res["sample_settings"] = SAMPLE_SETTINGS
    res.update(entry=e, frozen_sha256=sha256_text(frozen_text),
               excluded_docs={s: len(v) for s, v in excl.items()})
    return res
