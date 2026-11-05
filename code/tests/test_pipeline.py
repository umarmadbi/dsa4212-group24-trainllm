"""
Phase 0 correctness checks (E00). CPU only, ~1-2 minutes, no GPU credits.

    python code/tests/test_pipeline.py

Each test is one row of the E00 checklist. The tests build a tiny synthetic dataset
(token files, token byte lengths, tokenizer metadata) in a temporary directory, in the
same layout prepare_data.py writes, so they exercise the real training and evaluation code.
"""

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpb import evaluate_bpb  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from train import (Config, MixLoader, ShiftedSubsetLoader, WindowLoader, lr_at, run,  # noqa: E402
                   subset_window_ids)

V, EOT = 64, 63


def make_data(root: Path, n_train=60_000, n_val=6_000, seed=0):
    """Learnable synthetic text: a random 1st-order Markov chain over 63 tokens,
    documents of random length separated by EOT. Token byte lengths 1-4."""
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.full(V - 1, 0.05), size=V - 1)

    def stream(n):
        out, a = [], 0
        while len(out) < n:
            for _ in range(rng.integers(20, 200)):
                a = rng.choice(V - 1, p=trans[a]); out.append(a)
            out.append(EOT)
        return np.array(out[:n], dtype=np.uint16)

    for split, n in (("train", n_train), ("val", n_val)):
        p = root / "tokens" / "fineweb_edu" / f"{split}.bin"
        p.parent.mkdir(parents=True, exist_ok=True)
        stream(n).tofile(p)
    (root / "tokenizer").mkdir(parents=True, exist_ok=True)
    tb = rng.integers(1, 5, size=V).astype(np.int16); tb[EOT] = 0
    np.save(root / "tokenizer" / "token_bytes.npy", tb)
    (root / "tokenizer" / "meta.json").write_text(json.dumps(dict(vocab_size=V, eot_id=EOT)))


def tiny_cfg(**kw):
    base = dict(run_id="t", n_layer=2, n_head=2, n_embd=32, block_size=32, vocab_size=V,
                micro_batch=8, lr=3e-3, warmup_steps=5, compile=False, dtype="float32",
                curve_tokens=2048, eval_batch=16, log_every_s=1e9)
    base.update(kw)
    return Config(**base)


def test_initial_loss():
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=32, vocab_size=8192, bias=False))
    x = torch.randint(0, 8192, (8, 32))
    y = torch.randint(0, 8192, (8, 32))          # independent targets (tied embeddings favour y == x)
    _, loss = m(x, y)
    print(f"  initial loss {loss.item():.4f} vs ln(8192) = {math.log(8192):.4f}")
    assert abs(loss.item() - math.log(8192)) < 0.05


class Uniform(torch.nn.Module):
    """Predicts the uniform distribution; BPB is then known in closed form."""
    def forward(self, x):
        return torch.zeros(*x.shape, V), None


def test_bpb_hand_calculation(d: Path):
    toks = np.fromfile(d / "tokens/fineweb_edu/val.bin", dtype=np.uint16)
    tb = np.load(d / "tokenizer/token_bytes.npy")
    T = 32
    r = evaluate_bpb(Uniform(), toks, tb, T, 7, EOT, "cpu")
    nw = (len(toks) - 1) // T
    targets = toks[1:nw * T + 1]                    # every target scored once
    scored = targets[targets != EOT]                # EOT excluded
    expected = len(scored) * math.log(V) / (tb[scored].sum() * math.log(2))
    print(f"  bpb {r['bpb']:.6f} vs hand {expected:.6f}; EOT excluded {r['eot_targets_excluded']}")
    assert abs(r["bpb"] - expected) < 1e-6         # float32 cross-entropy rounding
    assert r["scored_tokens"] == len(scored)
    assert r["eot_targets_excluded"] == int((targets == EOT).sum())
    assert r["scored_bytes"] == int(tb[scored].sum())
    # batch size must not change the result (no per-batch averaging)
    r2 = evaluate_bpb(Uniform(), toks, tb, T, 1, EOT, "cpu")
    assert abs(r2["bpb"] - r["bpb"]) < 1e-9


def test_loader_covers_each_window_once():
    toks = np.arange(1001, dtype=np.uint16)
    L = WindowLoader(toks, T=10, seed=3)
    seen = []
    for _ in range(L.n_windows // 7 + 1):
        seen.extend(L.next_batch(7)[:, 0].tolist())
    first_pass = seen[:L.n_windows]
    assert sorted(first_pass) == list(range(0, L.n_windows * 10, 10))
    b = L.next_batch(1)
    assert torch.equal(b[0], torch.arange(b[0, 0].item(), b[0, 0].item() + 11))  # T+1 contiguous
    assert L.epoch >= 1


def test_lr_schedule():
    c = Config(lr=1.0, min_lr_ratio=0.1, warmup_steps=10, budget_s=100.0)
    assert abs(lr_at(c, 0, 0.0, None) - 0.1) < 1e-12
    assert abs(lr_at(c, 9, 1.0, None) - 1.0) < 1e-12
    assert abs(lr_at(c, 10, 5.0, 5.0) - 1.0) < 1e-12           # decay starts after warmup
    assert abs(lr_at(c, 500, 100.0, 5.0) - 0.1) < 1e-12        # exactly min LR at the cap
    assert abs(lr_at(c, 500, 52.5, 5.0) - 0.55) < 1e-12        # halfway: midpoint


def test_overfit_one_batch(d: Path):
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=V, bias=False))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    toks = np.fromfile(d / "tokens/fineweb_edu/train.bin", dtype=np.uint16)
    b = torch.from_numpy(toks[:8 * 33].astype(np.int64)).view(8, 33)
    for _ in range(300):
        _, loss = m(b[:, :-1], b[:, 1:].contiguous()); loss.backward(); opt.step(); opt.zero_grad()
    print(f"  loss after 300 steps on one batch: {loss.item():.4f}")
    assert loss.item() < 0.1


def test_budget_stop_and_learning(d: Path, out: Path):
    r = run(tiny_cfg(run_id="budget", budget_s=8.0, eval_every_frac=0.25), 0, str(d), str(out))
    print(f"  counted {r['counted_train_s']}s, {r['steps']} steps, curve points {len(r['curve'])}, "
          f"bpb {r['curve'][0]['val_bpb']:.3f} -> {r['final_val_bpb']:.3f}")
    assert 8.0 <= r["counted_train_s"] < 8.0 + 1.0
    assert r["final_val_bpb"] < r["curve"][0]["val_bpb"] - 0.3          # it learns
    assert len(r["curve"]) >= 5                                           # step 0 + periodic + final
    assert r["lr_log"] == [] or r["lr_log"][-1][2] < 3e-3
    assert abs(r["train_log"][-1]["lr"] - 3e-3 * 0.1) < 3e-3 * 0.2        # near min LR at the end
    # result must be plain JSON (it is unpickled on a computer without torch; E00 bug)
    def plain(o):
        if isinstance(o, dict):
            return all(type(k) is str and plain(v) for k, v in o.items())
        if isinstance(o, list):
            return all(plain(v) for v in o)
        return o is None or type(o) in (str, int, float, bool)
    assert plain(r), "result contains non-JSON types"


def test_resume_matches_continuous(d: Path, out: Path):
    common = dict(schedule="steps", max_steps=30, budget_s=1e9, ckpt_every_steps=10,
                  eval_every_frac=10.0)
    a = run(tiny_cfg(run_id="cont", **common), 1, str(d), str(out))
    os.environ["CRASH_AT_STEP"] = "25"
    try:
        run(tiny_cfg(run_id="crash", **common), 1, str(d), str(out))
    except SystemExit:
        pass
    finally:
        del os.environ["CRASH_AT_STEP"]
    ck = torch.load(out / "crash" / "ckpt_latest.pt", weights_only=False)
    assert ck["state"]["step"] == 20 and ck["state"]["counted"] > 0
    b = run(tiny_cfg(run_id="crash", **common), 1, str(d), str(out))
    wa = torch.load(out / "cont" / "final_model.pt", weights_only=False)["model"]
    wb = torch.load(out / "crash" / "final_model.pt", weights_only=False)["model"]
    maxdiff = max((wa[k] - wb[k]).abs().max().item() for k in wa)
    print(f"  resumed run: attempts {len(b['attempts'])}, resumed from step "
          f"{b['attempts'][-1]['resumed_from_step']}, max weight diff vs continuous {maxdiff:.2e}")
    assert b["steps"] == a["steps"] == 30
    assert len(b["attempts"]) == 2 and b["attempts"][1]["resumed_from_step"] == 20
    assert b["counted_train_s"] >= b["attempts"][1]["resumed_from_counted_s"]
    assert maxdiff < 1e-6
    assert abs(a["final_val_bpb"] - b["final_val_bpb"]) < 1e-6


def test_refuses_wrong_resume(d: Path, out: Path):
    try:
        run(tiny_cfg(run_id="crash", schedule="steps", max_steps=30, budget_s=1e9,
                     ckpt_every_steps=10, eval_every_frac=10.0), 2, str(d), str(out))
    except RuntimeError as e:
        assert "different config/seed" in str(e)
        return
    raise AssertionError("resumed a checkpoint with a different seed")


def test_crash_once_then_resume(d: Path, out: Path):
    """The config hook used for the Modal kill-and-resume test (D07): crash on attempt 1
    only, so the retry resumes and finishes; the result equals an uninterrupted run."""
    common = dict(schedule="steps", max_steps=30, budget_s=1e9, ckpt_every_steps=10,
                  eval_every_frac=10.0)
    a = run(tiny_cfg(run_id="cont2", **common), 3, str(d), str(out))
    cfg = tiny_cfg(run_id="crash2", test_crash_once_at_step=25, **common)
    try:
        run(cfg, 3, str(d), str(out))
        raise AssertionError("did not crash on attempt 1")
    except SystemExit:
        pass
    b = run(cfg, 3, str(d), str(out))                    # attempt 2: must not crash again
    assert b["steps"] == 30 and len(b["attempts"]) == 2
    assert b["attempts"][1]["resumed_from_step"] == 20
    assert abs(a["final_val_bpb"] - b["final_val_bpb"]) < 1e-6
    print(f"  crashed at step 25 on attempt 1, resumed from step 20, final BPB matches")


def test_divergence_guard(d: Path, out: Path):
    """A run whose loss becomes non-finite stops at once, is marked diverged, gets no BPB,
    and the result is still plain JSON (D07)."""
    t0 = __import__("time").time()
    r = run(tiny_cfg(run_id="nan", lr=float("inf"), budget_s=30.0, eval_every_frac=0.5),
            0, str(d), str(out))
    took = __import__("time").time() - t0
    print(f"  diverged at step {r['diverged_at_step']}, stopped after {r['counted_train_s']}s "
          f"counted ({took:.1f}s wall) of a 30 s budget")
    assert r["diverged"] is True and r["final_val_bpb"] is None
    assert r["diverged_at_step"] is not None and r["diverged_at_step"] <= 3
    assert r["counted_train_s"] < 5.0
    json.dumps(r)   # (lr=inf in this artificial test, so NaN-free output is not asserted)



def test_random_nested_subsets(d: Path, out: Path):
    """E04 (D23): random training subsets are nested, uniform over the split, the same for
    every run seed, and the loader reads only those windows, each once per pass."""
    n_tok, T = 100_001, 10
    small, big = subset_window_ids(n_tok, T, 2_000), subset_window_ids(n_tok, T, 20_000)
    assert len(small) == 200 and len(big) == 2_000
    assert set(small) <= set(big)                                  # nested
    assert small.max() > 0.8 * (n_tok // T) and small.min() < 0.2 * (n_tok // T)  # spread, not a prefix
    toks = np.arange(n_tok, dtype=np.uint32)
    L = WindowLoader(toks, T, seed=5, window_ids=small)
    starts = [int(L.next_batch(1)[0, 0]) for _ in range(200)]
    assert sorted(starts) == sorted(int(w) * T for w in small)     # each subset window once
    assert L.epoch == 1 and abs(L.passes() - 1.0) < 1e-9
    L2 = WindowLoader(toks, T, seed=6, window_ids=small)            # other seed: same windows,
    s2 = [int(L2.next_batch(1)[0, 0]) for _ in range(200)]          # different order
    assert sorted(s2) == sorted(starts) and s2 != starts
    # end to end: a subset run reports the subset as its unique data and counts passes
    r = run(tiny_cfg(budget_s=1e9, schedule="steps", max_steps=40, train_subset_tokens=3_200,
                     run_id="subset"),
            0, str(d), str(out))
    assert r["unique_train_tokens"] == 3_200 // 32 * 32
    assert r["passes_over_data"] > 1.0                              # 40 steps x 256 tokens > 3,200
    assert r["train_subset"]["windows"] == 3_200 // 32
    for bad in (dict(train_subset_tokens=10), dict(train_subset_tokens=3_200, train_tokens_limit=50_000)):
        try:
            run(tiny_cfg(budget_s=1e9, schedule="steps", max_steps=2, run_id="bad", **bad),
                0, str(d), str(out))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")

def test_all_config_files_load():
    """Every real config file loads with correct types (YAML 1.1 reads `1.0e9` as text;
    this bug crashed all 8 E00 benchmark runs on 2026-11-01)."""
    import dataclasses
    from config import Config, load_config
    root = Path(__file__).resolve().parents[2] / "configs"
    files = sorted(root.rglob("*.yaml"))
    assert files, "no config files found"
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    py = {"float": float, "int": int, "bool": bool, "str": str, "list": list}
    for f in files:
        c = load_config(str(f))
        for k, t in types.items():
            assert isinstance(getattr(c, k), py[t]), f"{f.name}: {k} is {getattr(c, k)!r}"
    assert load_config(None, {"ckpt_every_s": "1.0e9"}).ckpt_every_s == 1e9
    for bad in ({"budget_s": "abc"}, {"compile": "yes"}, {"budgt_s": 3}):
        try:
            load_config(None, bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"accepted bad config {bad}")
    print(f"  {len(files)} config files OK")


def test_no_pretrained_path():
    import model as M
    assert not hasattr(M.GPT, "from_pretrained")


def test_contamination_check():
    """E03 R3 (D25): the 13-gram index finds planted copies exactly like a brute-force set lookup,
    including matches that straddle the train chunk boundary, and skips documents < 13 tokens."""
    import contamination as C
    rng = np.random.default_rng(1)
    train = rng.integers(0, 60, 30_000).astype(np.uint16)
    docs = [rng.integers(0, 63, rng.integers(5, 150)).astype(np.uint16) for _ in range(40)]
    docs[2] = train[100:180].copy()                                   # full copy
    docs[5] = np.concatenate([train[4990:5030], rng.integers(0, 63, 40).astype(np.uint16)])  # spans chunk edge
    ev = np.concatenate([np.append(d, EOT) for d in docs]).astype(np.uint16)
    idx = C.EvalIndex({"x/val": ev}, EOT)
    fr = idx.doc_fractions("x/val", idx.mark(train, chunk=5_000))
    tr = {tuple(train[i:i + 13]) for i in range(len(train) - 12)}
    for d, f in zip(docs, fr):
        g = [tuple(d[i:i + 13]) for i in range(len(d) - 12)]
        assert (np.isnan(f) and not g) or abs(f - np.mean([x in tr for x in g])) < 1e-12
    assert fr[2] == 1.0 and 0.3 < fr[5] < 0.6
    s = C.summarise(fr)
    assert 2 in s["ge_50"]["doc_indices"] and s["docs_too_short"] == int(np.isnan(fr).sum())
    u = C.union(dict(by_train_corpus={"a": {"x/val": s}, "b": {"x/val": dict(s, ge_50=dict(doc_indices=[7]))}}))
    assert u["x/val"]["doc_indices"] == sorted(set(s["ge_50"]["doc_indices"]) | {7})


def test_rescore_reproduces_and_excludes(d: Path, out: Path):
    """rescore.py: a saved final model re-scores to exactly its stored BPB (float32, CPU), and removing
    documents equals scoring a file rebuilt by hand without them."""
    import rescore as R
    r = run(tiny_cfg(budget_s=1e9, schedule="steps", max_steps=20, run_id="rescore"), 0, str(d), str(out))
    res = R.rescore(str(out / "rescore"), str(d), ["fineweb_edu"], exclude={"fineweb_edu": [0, 3]}, device="cpu")
    e = res["sets"]["fineweb_edu"]
    assert abs(e["full"]["bpb"] - r["final_val"]["fineweb_edu"]["bpb"]) < 1e-9 and abs(e["diff_vs_stored"]) < 1e-9
    toks = np.fromfile(d / "tokens" / "fineweb_edu" / "val.bin", dtype=np.uint16)
    same, k0, r0 = R.rebuild_without(toks, EOT, [])
    assert np.array_equal(same, toks) and r0 == 0
    ends = np.flatnonzero(toks == EOT)
    keep = [toks[(ends[i - 1] + 1 if i else 0):ends[i] + 1] for i in range(len(ends)) if i not in (0, 3)]
    hand = np.concatenate(keep + [toks[ends[-1] + 1:]])
    t2, kept, removed = R.rebuild_without(toks, EOT, [0, 3])
    assert np.array_equal(t2, hand) and removed == 2 and kept == len(ends) - 2
    model, c, _ = R.load_model(out / "rescore" / "final_model.pt", "cpu")
    tb = np.load(d / "tokenizer" / "token_bytes.npy")
    assert abs(evaluate_bpb(model, hand, tb, c["block_size"], 16, EOT, "cpu")["bpb"] - e["excluded"]["bpb"]) < 1e-12


def _crash_resume_equal(d: Path, out: Path, tag: str, **kw):
    """Run continuously and with a simulated preemption at step 25; weights must match."""
    common = dict(schedule="steps", max_steps=30, budget_s=1e9, ckpt_every_steps=10,
                  eval_every_frac=10.0, **kw)
    a = run(tiny_cfg(run_id=f"{tag}_cont", **common), 1, str(d), str(out))
    os.environ["CRASH_AT_STEP"] = "25"
    try:
        run(tiny_cfg(run_id=f"{tag}_crash", **common), 1, str(d), str(out))
    except SystemExit:
        pass
    finally:
        del os.environ["CRASH_AT_STEP"]
    b = run(tiny_cfg(run_id=f"{tag}_crash", **common), 1, str(d), str(out))
    wa = torch.load(out / f"{tag}_cont" / "final_model.pt", weights_only=False)["model"]
    wb = torch.load(out / f"{tag}_crash" / "final_model.pt", weights_only=False)["model"]
    assert len(b["attempts"]) == 2 and max((wa[k] - wb[k]).abs().max().item() for k in wa) < 1e-6
    return a


def test_shifted_subset(d: Path, out: Path):
    """E04 F3 (D26): same subset text, window boundaries move every pass, seams marked by EOT,
    every stream token a target at most once per pass, resumable."""
    n_tok, T = 100_001, 10
    ids = subset_window_ids(n_tok, T, 2_000)
    toks = (np.arange(n_tok) % 50_000).astype(np.uint32)
    L = ShiftedSubsetLoader(toks, T, seed=3, window_ids=ids, eot=99_999)
    assert L.L == len(ids) * (T + 1) and (L.stream[T::T + 1] == 99_999).all()
    assert np.array_equal(np.sort(L.stream[L.stream != 99_999]), np.sort(toks[(ids * T)[:, None] + np.arange(T)].ravel()))
    offs, firsts = [], []
    for ep in range(3):
        x = L.next_batch(L.n_windows)                      # exactly one pass
        offs.append(L.offset); firsts.append(x[:, 0].numpy())
    starts0 = set(firsts[0].tolist()); starts1 = set(firsts[1].tolist())
    assert starts0 != starts1                             # boundaries differ between passes
    # coverage: within one pass, stream positions targeted are distinct
    L2 = ShiftedSubsetLoader(toks, T, seed=3, window_ids=ids, eot=99_999)
    o = L2.offset
    pos = ((o + L2._perm * T)[:, None] + np.arange(1, T + 1)) % L2.L
    assert len(np.unique(pos)) == pos.size >= L2.L - T
    s = L2.state(); L2.next_batch(5); L2.load_state(s)
    assert L2.offset == o
    r = _crash_resume_equal(d, out, "shift", train_subset_tokens=3_200, train_subset_shift=True)
    assert r["train_subset"]["shifted_windows"] and r["unique_train_tokens"] == 3_200 // 32 * 33


def test_corpus_mix(d: Path, out: Path):
    """E03 R8 (D26): batches mix corpora in the declared token shares (exact to one window), rows
    come from each corpus's random pool, and a mixed run resumes exactly."""
    rng = np.random.default_rng(9)
    p = d / "tokens" / "dclm" / "train.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    rng.integers(0, EOT, 50_000).astype(np.uint16).tofile(p)
    pools = [np.full((50, 5), 1), np.full((30, 5), 2)]
    M = MixLoader(pools, [0.6, 0.4], seed=0)
    for n in (1, 7, 64, 333):
        x = M.next_batch(n)
    drawn = M.state()["drawn"]
    tot = sum(drawn)
    assert abs(drawn[0] - 0.6 * tot) <= 1 and abs(drawn[1] - 0.4 * tot) <= 1
    a = _crash_resume_equal(d, out, "mix", corpus="mix", corpus_mix=["fineweb_edu=0.6", "dclm=0.4"],
                            mix_pool_tokens=20_000)
    cm = a["corpus_mix"]
    assert [c["corpus"] for c in cm["corpora"]] == ["fineweb_edu", "dclm"]
    assert abs(cm["realised_share"][0] - 0.6) < 0.01
    for bad in (dict(corpus="fineweb_edu", corpus_mix=["fineweb_edu=0.6", "dclm=0.4"]),
                dict(corpus="mix", corpus_mix=["fineweb_edu=0.7", "dclm=0.4"])):
        try:
            run(tiny_cfg(run_id="badmix", schedule="steps", max_steps=2, budget_s=1e9, **bad), 0, str(d), str(out))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_muon_optimizer(d: Path, out: Path):
    """E05 (D28): Muon gets exactly the hidden block matrices, AdamW the rest; one schedule drives both
    (Muon lr = muon_lr/lr x AdamW lr); the model learns; crash-resume is exact."""
    import train as TR
    cfg = tiny_cfg(optimizer="muon", muon_lr=6e-3)
    m = GPT(GPTConfig(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size,
                      vocab_size=cfg.vocab_size, dropout=0.0, bias=False))
    opt = TR.MuonAdamW(m, cfg, "cpu")
    assert opt.counts["muon_tensors"] == 4 * cfg.n_layer
    names = {n for n, p in m.named_parameters() if any(p is q for g in opt.muon.param_groups for q in g["params"])}
    assert all(n.startswith("transformer.h.") and n.endswith(TR.MUON_SUFFIXES) for n in names)
    assert opt.counts["muon_params"] + opt.counts["adamw_params"] == sum(p.numel() for p in m.parameters())
    assert abs(opt.muon.param_groups[0]["lr_scale"] - 2.0) < 1e-12
    a = _crash_resume_equal(d, out, "muon", optimizer="muon", muon_lr=6e-3)
    assert a["optimizer_info"]["name"] == "muon" and a["final_val_bpb"] < a["curve"][0]["val_bpb"] - 0.1   # learns
    for bad in (dict(optimizer="muon"), dict(optimizer="sgd")):
        try:
            from config import load_config
            load_config(None, bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    d, out = tmp / "data", tmp / "runs"
    make_data(d)
    tests = [("initial loss ~ ln(vocab)", test_initial_loss, ()),
             ("BPB evaluator = hand calculation", test_bpb_hand_calculation, (d,)),
             ("loader: each window once per pass", test_loader_covers_each_window_once, ()),
             ("LR schedule: warmup, cosine, min LR at cap", test_lr_schedule, ()),
             ("overfit one tiny batch", test_overfit_one_batch, (d,)),
             ("time budget stop + learning curve", test_budget_stop_and_learning, (d, out)),
             ("kill + resume == continuous run", test_resume_matches_continuous, (d, out)),
             ("refuse resume with other seed", test_refuses_wrong_resume, (d, out)),
             ("crash once (Modal test hook) + resume", test_crash_once_then_resume, (d, out)),
             ("divergence guard stops a NaN run", test_divergence_guard, (d, out)),
             ("random nested training subsets (E04)", test_random_nested_subsets, (d, out)),
             ("shifted windows per pass (E04 F3)", test_shifted_subset, (d, out)),
             ("corpus mixture: shares, pools, resume (E03 R8)", test_corpus_mix, (d, out)),
             ("Muon optimizer: parameter split, schedule, resume (E05)", test_muon_optimizer, (d, out)),
             ("13-gram contamination check = brute force (E03 R3)", test_contamination_check, ()),
             ("rescore reproduces stored BPB; document removal", test_rescore_reproduces_and_excludes, (d, out)),
             ("all config files load with correct types", test_all_config_files_load, ()),
             ("no pretrained-weights code path", test_no_pretrained_path, ())]
    failed = 0
    for name, fn, args in tests:
        try:
            fn(*args)
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    shutil.rmtree(tmp)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
