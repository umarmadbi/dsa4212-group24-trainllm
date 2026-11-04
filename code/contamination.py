"""
13-gram contamination check (DECLARATIONS section 3; rule R3 of E03, made precise in D25).

Plain language: web crawls overlap. A validation or test document might also sit, word for word or
almost, inside a training set (a copied page, a mirror, a quoted Wikipedia paragraph). A model would
then score well on it by memory, not by skill. We look for long shared passages: every run of 13
consecutive tokens (a "13-gram", ~10 words) of every evaluation document is looked up in the FULL train
split of each training corpus. A document is **contaminated by a corpus if >= 50 % of its 13-grams
occur somewhere in that corpus's train split** (declared D25); counts at >= 10 % and >= 80 % and the
whole distribution are reported as well.

Method (exact up to 64-bit hash collisions, expected < 0.01 false matches over the whole check):
* Tokens are our 8k BPE ids (the files training reads). Evaluation files are documents each followed by
  <|endoftext|> (EOT); a 13-gram never spans a document boundary on the evaluation side, and documents
  shorter than 13 tokens are skipped (counted).
* Each 13-gram -> 64-bit polynomial hash. The set of all evaluation 13-gram hashes (val + test of all four
  corpora, ~40M) is sorted once; each train split is streamed in chunks, hashed the same way, and looked up
  with a binary search. A hit marks that evaluation 13-gram as "present in this training corpus".
* Per document: fraction of its 13-grams present. Train 13-grams that span an EOT cannot match an
  evaluation 13-gram (those never contain EOT), so they are harmless.

The training text a model actually reads is a ~12 % random sample of the train split, so checking the
full split is conservative (it over-counts what any one run could have memorised).

Use on Modal (CPU only, cents; data must be on that workspace's Volume, any verified workspace):
    modal run --detach modal_app.py::contamination          (starts it; survives a dropped connection)
    modal run modal_app.py::contamination --fetch-only      (later: writes results/contamination.json)
Locally (tests, or a machine holding the token files):
    python code/contamination.py --data_dir <dir with tokens/ and tokenizer/> --out results/contamination.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

N = 13
P = np.uint64(0x9E3779B97F4A7C15)       # odd 64-bit multiplier (wrap-around arithmetic mod 2^64)
THRESHOLDS = (0.10, 0.50, 0.80)          # 0.50 is the declared one (D25)
DECLARED = 0.50
TRAIN_CORPORA = ("fineweb_edu", "fineweb", "dclm")
EVAL_CORPORA = ("fineweb_edu", "fineweb", "dclm", "wikipedia")
CHUNK = 50_000_000


def ngram_hashes(tokens: np.ndarray, n: int = N) -> np.ndarray:
    """Hash of every window tokens[i:i+n], i = 0..len-n (uint64, wraps mod 2^64)."""
    t = tokens.astype(np.uint64)
    m = len(t) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * P + t[j:j + m] + np.uint64(1)    # +1 so token 0 is not a no-op
    return h


def split_docs(tokens: np.ndarray, eot: int) -> list[np.ndarray]:
    """Documents of an evaluation file (each document is followed by EOT), in file order."""
    ends = np.flatnonzero(tokens == eot)
    docs, start = [], 0
    for e in ends:
        docs.append(tokens[start:e]); start = e + 1
    if start < len(tokens):              # tail without EOT (not expected; kept for safety)
        docs.append(tokens[start:])
    return docs


class EvalIndex:
    """All 13-gram hashes of every evaluation document, sorted and de-duplicated, plus for each
    document the positions of its 13-grams in that sorted array."""

    def __init__(self, eval_sets: dict[str, np.ndarray], eot: int):
        self.names, per_doc, allh = [], {}, []
        for name, toks in eval_sets.items():
            docs = split_docs(toks, eot)
            per_doc[name] = [ngram_hashes(d) for d in docs]
            allh.extend(per_doc[name])
            self.names.append(name)
        self.U = np.unique(np.concatenate(allh)) if allh else np.zeros(0, np.uint64)
        self.doc_pos = {name: [np.searchsorted(self.U, h) for h in hs] for name, hs in per_doc.items()}

    def mark(self, train_tokens: np.ndarray, chunk: int = CHUNK, log=None) -> np.ndarray:
        """Boolean array over U: does this 13-gram occur anywhere in train_tokens?"""
        found = np.zeros(len(self.U), dtype=bool)
        if len(self.U) == 0:
            return found
        n_tok = len(train_tokens)
        for start in range(0, max(1, n_tok - N + 1), chunk):
            seg = np.asarray(train_tokens[start:min(n_tok, start + chunk + N - 1)])
            h = ngram_hashes(seg)
            pos = np.searchsorted(self.U, h)
            np.minimum(pos, len(self.U) - 1, out=pos)
            hit = self.U[pos] == h
            found[pos[hit]] = True
            if log:
                log(f"    {min(n_tok, start + chunk) / 1e6:,.0f}M / {n_tok / 1e6:,.0f}M tokens, "
                    f"{int(found.sum()):,} eval 13-grams found so far")
        return found

    def doc_fractions(self, name: str, found: np.ndarray) -> np.ndarray:
        """Per document: fraction of its 13-grams found (NaN for documents < 13 tokens)."""
        return np.array([found[p].mean() if len(p) else np.nan for p in self.doc_pos[name]])


def summarise(fr: np.ndarray) -> dict:
    ok = fr[~np.isnan(fr)]
    out = dict(docs=int(len(fr)), docs_scored=int(len(ok)), docs_too_short=int(np.isnan(fr).sum()),
               mean_fraction=float(ok.mean()) if len(ok) else None,
               quantiles={str(q): float(np.quantile(ok, q)) for q in (0.5, 0.9, 0.99, 1.0)} if len(ok) else {})
    for t in THRESHOLDS:
        idx = [int(i) for i in np.flatnonzero(np.nan_to_num(fr, nan=0.0) >= t)]
        out[f"ge_{int(t * 100)}"] = dict(count=len(idx), share_of_docs=len(idx) / max(1, len(fr)),
                                         doc_indices=idx)
    return out


def run_check(data_dir: str, train_corpora=TRAIN_CORPORA, splits=("val", "test"), log=print,
              chunk: int = CHUNK) -> dict:
    d = Path(data_dir)
    eot = json.loads((d / "tokenizer" / "meta.json").read_text())["eot_id"]
    eval_sets = {}
    for c in EVAL_CORPORA:
        for s in splits:
            f = d / "tokens" / c / f"{s}.bin"
            if f.exists():
                eval_sets[f"{c}/{s}"] = np.fromfile(f, dtype=np.uint16)
    t0 = time.time()
    idx = EvalIndex(eval_sets, eot)
    log(f"evaluation index: {len(eval_sets)} files, {len(idx.U):,} distinct 13-grams ({time.time() - t0:.0f}s)")
    res = dict(n=N, declared_threshold=DECLARED, thresholds=list(THRESHOLDS), unit="8k BPE token 13-grams",
               train_reference="full train split of each training corpus", by_train_corpus={})
    for tc in train_corpora:
        f = d / "tokens" / tc / "train.bin"
        if not f.exists():
            log(f"skip {tc}: no train.bin"); continue
        tr = np.memmap(f, dtype=np.uint16, mode="r")
        log(f"train corpus {tc}: {len(tr):,} tokens")
        t1 = time.time()
        found = idx.mark(tr, chunk=chunk, log=log)
        res["by_train_corpus"][tc] = {name: summarise(idx.doc_fractions(name, found)) for name in eval_sets}
        res["by_train_corpus"][tc]["_seconds"] = round(time.time() - t1, 1)
    res["union_declared"] = union(res)
    return res


def union(res: dict, key: str = "ge_50") -> dict:
    """Per evaluation file: documents contaminated by ANY training corpus at the declared threshold.
    These are the documents removed for EVERY arm in the E03 R3 sensitivity (paired comparison)."""
    out = {}
    for tc, by in res["by_train_corpus"].items():
        for name, s in by.items():
            if name.startswith("_"):
                continue
            u = out.setdefault(name, dict(docs=s["docs"], doc_indices=set()))
            u["doc_indices"] |= set(s[key]["doc_indices"])
    return {name: dict(docs=u["docs"], count=len(u["doc_indices"]),
                       share_of_docs=len(u["doc_indices"]) / max(1, u["docs"]),
                       over_1_percent=len(u["doc_indices"]) > 0.01 * u["docs"],
                       doc_indices=sorted(u["doc_indices"])) for name, u in out.items()}


def report(res: dict) -> str:
    lines = [f"13-gram contamination (declared: >= {int(DECLARED * 100)} % of a document's 13-grams in a train split)"]
    for tc, by in res["by_train_corpus"].items():
        lines.append(f"  train corpus {tc} ({by.get('_seconds', '?')} s):")
        for name, s in by.items():
            if name.startswith("_"):
                continue
            lines.append(f"    {name:18s} docs {s['docs']:5d} (short {s['docs_too_short']}) mean frac "
                         f"{(s['mean_fraction'] or 0):.4f}  >=10% {s['ge_10']['count']:4d}  "
                         f">=50% {s['ge_50']['count']:4d} ({s['ge_50']['share_of_docs']:.2%})  >=80% {s['ge_80']['count']:4d}")
    lines.append("  union over training corpora at the declared threshold (removed for every arm if > 1 %):")
    for name, u in res["union_declared"].items():
        lines.append(f"    {name:18s} {u['count']:4d} of {u['docs']} ({u['share_of_docs']:.2%})"
                     + ("  -> > 1 %: E03 R3 sensitivity re-score REQUIRED" if u["over_1_percent"] and name.endswith("/val") else ""))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default="results/contamination.json")
    a = p.parse_args(argv)
    res = run_check(a.data_dir)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(report(res))


if __name__ == "__main__":
    main()
