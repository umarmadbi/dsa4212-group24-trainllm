# tools/

Helpers that run outside the training code. Kept out of `code/` on purpose: the training launcher
(`modal_app.py::launch`) fingerprints `code/`, `configs/`, `modal_app.py` and `requirements.txt`, and these
tools were added while Phase 4 was running, so Phase 4's later rounds keep the same code fingerprint (D31).

| File | What it does |
|---|---|
| `prepare_vocab.py` | Builds the data for another vocabulary size (Phase 5): trains the byte-level BPE tokenizer with exactly the Phase 0 recipe, then tokenizes FineWeb-Edu train/val/test and the val/test splits of FineWeb, DCLM and Wikipedia. Writes **only** under `/data/vocab/v<size>/` (never the baseline `/data/tokenizer`, `/data/tokens`, `/data/manifest.json`), builds in a temporary folder and renames it only when complete, refuses to overwrite. Reuses the raw text already on the Volume (no download) and checks every split has the same documents and bytes as the baseline |
| `vocab_app.py` | Modal app (CPU only) around it. `build` starts one container per size plus a check that the builder's 8,192 tokenizer is byte-identical to the Phase 0 one; `fetch` copies each variant's manifest and tokenizer into `data/vocab/v<size>/` and prints bytes per token; `verify` re-hashes every variant file **and every baseline file** |
| `test_prepare_vocab.py` | CPU test on fake data: same vocabulary = byte-identical to `code/prepare_data.py`; baseline files unchanged; same documents per split; refuses overwrite, a different tokenizer sample, and paths outside `vocab/` |

Usage (from the repo root, on the workspace that will run Phase 5):

```
modal run --detach tools/vocab_app.py::build      # ~30-60 min on Modal (4096, 16384, 32768 + 8192 check)
modal run tools/vocab_app.py::fetch               # afterwards: must print "8,192 REPRODUCTION CHECK: PASSED"
modal run tools/vocab_app.py::verify              # must print "VOCAB DATA CHECK PASSED" and baseline 13/13 unchanged
python tools/test_prepare_vocab.py                # local test (needs tokenizers, numpy)
```
