# Training small language models in 60 minutes on one GPU

DSA4212 Group 24 · [Authors](AUTHORS.md) · [Report](report/) · [Project overview](docs/OVERVIEW.md) · [How to reproduce](docs/REPRODUCE.md)

**If you can train a language model from scratch for only one hour on one GPU, how should you spend that hour?**

We trained GPT-style models (7M to 65M parameters) from random weights, each for exactly 60 minutes on one NVIDIA L4,
and changed one thing at a time: model size, training data, data repetition, optimiser, model shape and vocabulary
size. The two changes that helped most, the Muon optimiser and a 16,384-token vocabulary, add up almost exactly when
combined: the final model predicts held-out text **0.041 bits per byte better** than our tuned baseline, and 0.050 with
a wider model shape.

## Declared setup (fixed before any experiment)

| | |
|---|---|
| **Compute budget** | 60 minutes of counted training time per run on 1 x NVIDIA L4 (bf16), rented on [Modal](https://modal.com). Learning-rate pilots and tokenizer training are excluded, as the brief allows |
| **Initialisation** | Random weights only. GPT architecture from [nanoGPT](https://github.com/karpathy/nanoGPT); our own byte-level BPE tokenizer; no pretrained weights, embeddings or other components |
| **Evaluation** | Bits per byte (BPB) on held-out FineWeb-Edu text: comparable across tokenizers, lower is better. Validation data for every decision; the test split scored once, at the end, for 57 models frozen in advance |

Full rules and every later amendment: [docs/DECLARATIONS.md](docs/DECLARATIONS.md).

## Main result

Test set, FineWeb-Edu, mean of 3 training seeds unless noted. Seed-to-seed noise is about 0.0015 BPB; the declared
threshold for a real difference between two 3-seed groups is 0.0091.

| Model | Test BPB | vs baseline |
|---|---|---|
| Baseline: AdamW, 11 layers x 576 wide (49M parameters), 8,192-token vocabulary | 1.1994 | |
| Muon optimiser instead of AdamW | 1.1731 | -0.026 |
| 16,384-token vocabulary instead of 8,192 | 1.1834 | -0.016 |
| 6 layers x 768 instead of 11 x 576, same size (1 seed) | 1.1814 | -0.018 |
| **Recipe RB: Muon + 16k vocabulary (declared final)** | **1.1583** | **-0.041** |
| **Recipe RA: Muon + 16k vocabulary + 6 x 768 (best measured)** | **1.1494** | **-0.050** |

Other findings: within one hour the best model is much larger than Chinchilla's rule suggests (36M-65M parameters,
4-11 tokens per parameter instead of ~20); DCLM text beats FineWeb-Edu on the average over four evaluation sets, and a
60/40 mix of the two beats both; repeating the same data becomes harmful between 2 and 4 passes; a deep, narrow model
(19 x 448) is worse. Every validation conclusion held on the test set. Details: [docs/OVERVIEW.md](docs/OVERVIEW.md).

## Quick start

You need **Python 3.12 or newer** (3.11 cannot install the pinned packages), **git**, and a [Modal](https://modal.com)
account with credit. All GPU and data work runs on Modal; your computer only launches jobs and summarises results.
Clone the repository with git rather than downloading a ZIP: every run records the git commit it was launched from.
Commands are for Windows PowerShell; macOS/Linux differences are in the comments.

**1. Install (once, ~2 minutes)**

```powershell
git clone https://github.com/umarmadbi/dsa4212-group24-trainllm.git
cd dsa4212-group24-trainllm
py -3.12 -m venv .venv                     # macOS/Linux: python3.12 -m venv .venv
.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
pip install -r setup/requirements-dev.txt  # modal, pyyaml, scipy
modal setup                                # log in to Modal (opens a browser)
modal run modal_app.py::gpu_check          # ~1 min: must print an NVIDIA L4
```

**2. Check the code and the saved results on your computer (no Modal, free)**

```powershell
pip install torch numpy pyyaml tokenizers
python code/tests/test_pipeline.py         # ~2-5 min on CPU; last line must be: 22/22 passed
python code/summarize.py E02               # rebuilds the model-size table from results/runs/
```

**3. Build the data on your Modal account (once, ~1 hour, ~$0.60)**

```powershell
modal run --detach modal_app.py::rebuild_data   # downloads the pinned dataset versions, builds tokens + tokenizer
modal run modal_app.py::verify_data             # after it finishes: must print DATA CHECK PASSED
```

**4. Reproduce a run**

```powershell
modal run --detach modal_app.py::launch --configs configs/E02/N49.yaml --seeds 0 --kind final --member <your-name>
```

This retrains the baseline for 60 minutes (~$0.83). With `--detach` it keeps running if you close the terminal; fetch the
result afterwards with `modal run modal_app.py::recover --run-ids "E02-N49__s0" --configs configs/E02/N49.yaml --kind final`.
The result is written to `results/runs/E02-N49__s0.json` (our original is moved to `results/runs/_replaced/` and stays in
git). Our three seeds scored 1.1936, 1.1961 and 1.1937 BPB; a new run should land within about 0.01 of these (single-run
noise is ~0.006 BPB; runs on cloud GPUs are not bit-identical because a faster machine reads more text in the hour).

The full guide (every experiment, the vocabulary variants, evaluation checks, expected values, and a
**troubleshooting** table for common errors) is [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Repository layout

Every folder has a README that describes each file in it.

| Folder / file | Contents |
|---|---|
| [`modal_app.py`](modal_app.py) | The entry point for every job on Modal: training runs, data preparation and checks, re-scoring, test scoring |
| [`code/`](code/) | Training loop, model, metric, data preparation, evaluation, results summaries, tests |
| [`configs/`](configs/) | One YAML file per training run, in one folder per experiment (E00-E08), and the test-set lock |
| [`data/`](data/) | Tokenizers and the data manifest (versions, sizes, hashes); the token files live on Modal |
| [`tools/`](tools/) | Builder for the vocabulary-size variants (experiment E07) |
| [`results/`](results/) | Every run's result file, the compute ledger, re-scores, contamination check, test scores |
| [`docs/`](docs/) | Declared rules, project overview, reproduction guide, experiment write-ups, decision log, lab log |
| [`report/`](report/) | The report source (LaTeX), bibliography and figures |
| [`setup/`](setup/) | Package list for the laptop that launches jobs |
| `requirements.txt` | Exact package versions inside the Modal container (used by every run) |

## Team

Group 24: Jeon Hyewon, Jeon Jinkyung, Joshua Han Rui Chou, Muhammad Umar Bin Madbi. Student numbers in [AUTHORS.md](AUTHORS.md).
