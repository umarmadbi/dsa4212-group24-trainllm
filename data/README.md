# data/

The tokenizer and the record of exactly which data we use. **The training text itself is not here**: it is about
15 GB of token files, stored in a Modal Volume called `small-llm-data` that the training jobs read directly.

| File | What it is |
|---|---|
| `tokenizer.json` | Our byte-level BPE tokenizer (8,191 BPE tokens + 1 end-of-text id = **8,192**), trained on train splits only. Contains no neural network |
| `tokenizer_meta.json` | Tokenizer facts: vocabulary size, the end-of-text id (8191), the text it was trained on (200 MB from each training corpus), and SHA-256 hashes of the tokenizer files |
| `manifest.json` | **The data record.** For every corpus: the pinned Hugging Face version (commit hash), the source files used, document and token counts per split, bytes per token, duplicates removed, and the **SHA-256 hash of every token file**. Written by `code/prepare_data.py` |
| [`vocab/`](vocab/) | The tokenizers and data records of the three vocabulary-size variants used in E07 and E08 (4,096, 16,384, 32,768 tokens). See [vocab/README.md](vocab/README.md) |

Do not edit these files. The hashes in `manifest.json` prove that validation and test never changed, and
`.gitattributes` keeps their line endings fixed so the hashes also match on Windows.

## The corpora

| Corpus | Source (Hugging Face) | Role | Train tokens | Validation / test |
|---|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu`, sample-10BT | **Main training corpus** | 2.45B | ~10 MB of text each (~2.6M tokens) |
| FineWeb | `HuggingFaceFW/fineweb`, sample-10BT | Data-quality comparison (unfiltered web) | 2.53B | ~10 MB each |
| DCLM | `mlfoundations/dclm-baseline-1.0` | Data-quality comparison (different quality filter) | 2.57B | ~10 MB each |
| Wikipedia | `wikimedia/wikipedia` (English) | Out-of-domain **evaluation only** | none | ~10 MB each |

Each training corpus has more tokens than the fastest model can read in 60 minutes (~1.5B), so no run in the main
experiments ever sees the same text twice.

## What is on Modal (`small-llm-data` Volume)

```
/data/
├── tokenizer/
│   ├── tokenizer.json, meta.json
│   └── token_bytes.npy        bytes of text per token id (the BPB denominator)
├── tokens/
│   ├── fineweb_edu/  train.bin  val.bin  test.bin
│   ├── fineweb/      train.bin  val.bin  test.bin
│   ├── dclm/         train.bin  val.bin  test.bin
│   └── wikipedia/               val.bin  test.bin
└── vocab/v<V>/       the same layout for each vocabulary variant (E07), built by tools/vocab_app.py
```

`.bin` files are token ids stored as `uint16`, documents joined with the end-of-text id. Training runs read the
whole train file into memory before the clock starts.

## Splits and the test-set lock

- Documents (not tokens) were split into train / validation / test with a fixed seed (1234).
- Any training document identical to a validation or test document of **any** corpus was removed (FineWeb-Edu is a
  subset of FineWeb, so this matters).
- **Validation** is used for every choice we make. **Test** is used once, at the very end, only for configurations
  frozen in advance.

## Using another group member's Modal account

Each Modal account (workspace) has its own Volume, so a new workspace must build its own copy of the data first.
With your CLI profile pointed at that workspace (`modal profile current` to check):

```powershell
modal run modal_app.py::rebuild_data     # CPU only, ~1 hour, ~$0.60; then checks every hash
modal run modal_app.py::verify_data      # the hash check on its own (a few minutes)
```

`rebuild_data` uses the sizes **and the exact Hugging Face dataset commits** recorded in `manifest.json` (not the
datasets' latest versions), then checks that every token file and the tokenizer match `manifest.json`.
[DECLARATIONS.md](../docs/DECLARATIONS.md) requires `DATA CHECK PASSED` on a workspace before any run there counts.
