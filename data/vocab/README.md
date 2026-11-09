# data/vocab/

The tokenizers and data records of the **vocabulary-size variants** (experiment E07, and the 16k vocabulary in the E08
recipes). Built on Modal by [`tools/vocab_app.py`](../../tools/vocab_app.py) with exactly the Phase 0 tokenizer recipe and
training sample; only the number of BPE merges differs. Every variant tokenizes the same documents and splits as the
8,192-token baseline in [`data/`](../). As with the baseline, the token files themselves are stored on Modal
(`/data/vocab/v<V>/` on the `small-llm-data` Volume) and only the records are kept here.

| Folder | Vocabulary | Used by |
|---|---|---|
| `v4096/` | 4,096 tokens | E07 |
| `v16384/` | 16,384 tokens | E07, both E08 recipes and the declared final model |
| `v32768/` | 32,768 tokens | E07 |

Each folder holds the same three files:

| File | What it is |
|---|---|
| `tokenizer.json` | The byte-level BPE tokenizer of that size (vocabulary - 1 BPE tokens + 1 end-of-text id) |
| `meta.json` | Vocabulary size, end-of-text id, the text it was trained on (200 MB from each training corpus) and SHA-256 hashes of the tokenizer files |
| `manifest.json` | Document and token counts per split and the SHA-256 hash of every token file, plus the hash of the baseline manifest it was derived from. File paths were normalised to `/data/...` after the build (noted in the file) |

Do not edit these files: `modal run tools/vocab_app.py::verify` re-hashes every variant file on Modal against them, and
every E07/E08 result file records the hashes of the data it trained on.
