# code/tests/

Checks that the training and evaluation code is correct **before** any GPU money is spent. They run on CPU with a
tiny synthetic dataset, in about 2 minutes, and use the real `train.py`, `model.py` and `bpb.py`.

Run from the repo root (needs `pip install torch numpy pyyaml`):

```powershell
python code/tests/test_pipeline.py
```

Expected last line: `22/22 passed`. Run it after any change to `code/` and before committing.

| File | What it does |
|---|---|
| `test_pipeline.py` | The 22 tests below. Each one is a row of the correctness checklist in the Phase 0 records |

| Test | What it proves |
|---|---|
| initial loss ~ ln(vocab) | A freshly initialised model is not already biased (loss = 9.01 for 8,192 tokens) |
| BPB evaluator = hand calculation | `bpb.py` gives exactly the hand-computed bits per byte; end-of-text tokens are excluded |
| loader: each window once per pass | Every training window is seen exactly once per pass (needed for the repetition experiment) |
| LR schedule | Warm-up, then cosine decay reaching the minimum exactly at the time cap |
| overfit one tiny batch | The model can learn (loss -> ~0 on one batch) |
| time budget stop + learning curve | Training stops at the budget, the clock pauses during evaluation, the model learns, the result is plain JSON |
| kill + resume == continuous run | A run killed at step 25 and resumed gives weights identical to an uninterrupted run |
| refuse resume with other seed | A checkpoint is never resumed with the wrong config or seed |
| crash once + resume | The hook used by the Modal kill-and-resume test crashes only on the first attempt |
| divergence guard | A run whose loss becomes NaN stops at once and is marked `diverged` |
| random nested training subsets (E04) | Repetition subsets are nested, spread over the whole train split (not a prefix), the same for every seed, and each subset window is read once per pass (D23) |
| shifted windows per pass (E04 F3) | Same subset text, window boundaries move every pass, seams marked by end-of-text, each token a target at most once per pass, exact resume after a crash (D26) |
| corpus mixture (E03 R8) | Batches follow the declared token shares exactly (to one window), rows come from each corpus's pool, exact resume after a crash, bad mixes refused (D26) |
| Muon optimizer (E05) | Muon gets exactly the hidden block matrices, AdamW the rest, one schedule drives both, the model learns, exact resume (D28) |
| 13-gram contamination check (E03 R3) | `contamination.py` finds shared 13-grams exactly like a brute-force lookup, also across chunk boundaries; union over training corpora (D25) |
| vocabulary variant data (E07) | `data_variant` reads only `<data>/vocab/v<V>/`, refuses a mismatched/unsupported variant, never falls back to the baseline files, cuts the curve subset to the baseline's text, hashes the variant files; rescore uses the variant; baseline results get no new fields (D36) |
| sliding-window BPB (E07) | The overlapping-window diagnostic equals the declared BPB exactly at stride = context and scores every target once (D36) |
| combined recipe (E08) | Muon + a vocabulary variant + a different shape, never run together before, train through the same code: Muon gets 4 x n_layer matrices, the variant's files are used and hashed, the model learns, a preemption resumes exactly, rescore reads the variant; the AdamW-part LR is a separate knob (D42) |
| test lock (D42) | `evaluate_test.py` refuses a draft list and unlisted runs; scores the TEST split (not val) of the model's own vocabulary; removes the listed contaminated test documents; adds the overlapping-window score; text samples decode with the real tokenizer; every run in the real `configs/frozen_configs.yaml` has a result file recorded on the listed workspace |
| rescore reproduces stored BPB | `rescore.py` re-scores a saved model to exactly its stored BPB, and leaving documents out equals scoring a hand-rebuilt file (D25) |
| all config files load | Every YAML file in `configs/` loads with the right types (a YAML quirk crashed the first benchmarks) |
| no pretrained-weights code path | `model.py` cannot load GPT-2 weights |

The tests run on CPU, so they cannot catch GPU-only problems. Two bugs of that kind were found on Modal instead (see
the E00 and E01 records): the resume crash on GPU, and container reuse skipping the compile warm-up.
