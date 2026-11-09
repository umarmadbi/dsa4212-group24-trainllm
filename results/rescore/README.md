# results/rescore/

Validation re-scores of saved final models, written by `modal run modal_app.py::rescore` (code: `code/rescore.py`). Each
re-score first checks that it reproduces the model's stored validation score exactly, then adds the extra scores. The
file name is `<run>__<tag>.json`. These are diagnostics; no decision used a re-score in place of the declared score.

| Tag | Files | What it is | Why |
|---|---|---|---|
| `nocontam` | 12 | Validation BPB without the documents flagged by the 13-gram contamination check (`../contamination.json`), on all four validation sets | E03 rule R3: does the data-quality ranking survive removing contaminated documents? (D25) |
| `full` | 1 | The validation sets missing from `E01-N26_lr4e-3_60min__s0` (it was trained before the four-set evaluation was added) | So the reused N26 seed 0 can be compared on every set (D43) |
| `slide128` | 14 | FineWeb-Edu validation BPB with overlapping windows (stride 128: every scored token has at least 384 tokens of context) | E07 check (D39): how much of the vocabulary effect comes from the scoring window: E07 60-min models and the baseline (D39), the other two 16k seeds (D43), the six E08 finals (D46) |

`python code/summarize.py E07` and `E08` print the overlapping-window comparison.
