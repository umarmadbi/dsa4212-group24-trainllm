# setup/

Packages for the computer that **launches** jobs (your laptop). Training itself runs on Modal, inside a container whose
packages are pinned separately in [`../requirements.txt`](../requirements.txt) (that file stays at the repository root
because `modal_app.py` reads it from there and records its hash in every result).

| File | What it is |
|---|---|
| `requirements-dev.txt` | `modal`, `pyyaml` and `scipy` with their dependencies, at the versions we used. Enough to launch runs, run `code/summarize.py` and the post-hoc fits (`code/fit_scaling_law.py`, `code/fit_repetition.py`) |

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1     # macOS/Linux: source .venv/bin/activate
pip install -r setup/requirements-dev.txt
modal setup                                         # log in to your Modal account
```

The CPU correctness tests (`code/tests/test_pipeline.py`) and the vocabulary-builder test (`tools/test_prepare_vocab.py`)
also need `torch`, `numpy` and `tokenizers`: `pip install torch numpy tokenizers`.
