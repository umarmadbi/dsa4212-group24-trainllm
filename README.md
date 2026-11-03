# DSA4212 Group 24: Training small language models under computational constraints

**If you get one hour on one GPU to train a language model from scratch, how should you spend that hour?**

We train small GPT-style models (7M-65M parameters) from random initialisation, each for exactly **60 minutes on one
NVIDIA L4 GPU** (rented on [Modal](https://modal.com)). We then ask what gives the best model for that fixed time:
a bigger model, more or better data, or training tricks. Models are scored by **bits per byte (BPB)** on held-out
text (lower is better).

**[The fixed rules (DECLARATIONS.md)](DECLARATIONS.md)**: budget, data, tokenizer, evaluation and seeds, committed
before any experiment. Read this first.

---

## Status

| Phase | What it does | Status |
|---|---|---|
| 0 | Data, tokenizer, training code, tests, speed benchmarks | Done |
| 1 | Tune the learning rate for each model size (15-min runs) + a 60-min check | Done: 2e-3 or 4e-3 for every size (DECLARATIONS Amendments) |
| 2 | Main experiment: which model size is best in 60 minutes? | **Next** |
| 3 | Data quality (FineWeb-Edu vs FineWeb vs DCLM) and repetition | Planned |
| 4 | Training tricks: batch size, schedule, optimiser (pick 2-3) | Planned |
| 5 | Vocabulary size | If time |
| 6 | Final: best recipe vs baseline, 3 seeds each, test set used once | Planned |

---

## Quick start (Windows PowerShell; the same commands work on macOS/Linux)

```powershell
# 1. install (once)
git clone https://github.com/umarmadbi/dsa4212-group24-trainllm.git
cd dsa4212-group24-trainllm
python -m venv .venv
.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt        # modal + pyyaml: enough to launch runs
modal setup                                # log in to your Modal account (opens a browser)

# 2. check Modal gives you an L4 (~1 min, ~1 cent)
modal run modal_app.py::gpu_check

# 3. launch one training run (configs are in configs/; see configs/README.md)
modal run modal_app.py::launch --configs configs/E01/N7_lr2e-3.yaml --seeds 0 --kind pilot

# 4. summarise the results of an experiment
python code/summarize.py E01
```

When a run finishes, `launch` writes `results/runs/<run_id>.json` and adds one row to `results/compute_ledger.csv`
on your computer. Keep the terminal open until it prints `done ...`.

**Optional: run the correctness tests on your laptop** (CPU only, ~2 min, needs `pip install torch numpy`):
`python code/tests/test_pipeline.py`. Expect `12/12 passed`.

### Rules everyone must follow

1. **Commit before you launch.** `launch` refuses to start if `code/`, `configs/`, `modal_app.py` or
   `requirements.txt` have uncommitted changes. Every result records the commit it ran from.
2. **Never edit or delete anything in `results/`.** A rerun keeps the old result (`results/runs/_replaced/`).
3. **Never use the test set to choose anything.** All choices use the validation split; the test set is used once,
   at the end, for frozen configurations only (DECLARATIONS.md section 5).
4. **One config file per run.** To change a setting, make a new config with a new `run_id`; don't edit a config
   that has already been run.
5. **On Windows, list configs with commas.** `--configs "configs/E01/*.yaml"` fails because the Modal command
   expands the `*` itself. Use `--configs "configs/E01/a.yaml,configs/E01/b.yaml"`.
6. **Launching from your own Modal account?** Pass `--member <your name>`. Your account first needs its own copy
   of the data (see [data/README.md](data/README.md)).

---

## How one run works

```
configs/E01/N26_lr2e-3.yaml   (all settings for the run)
        |
        |  modal run modal_app.py::launch --configs ... --seeds 0
        v
Modal starts a fresh container with one L4 GPU
  -> loads the training tokens into memory (not timed)
  -> builds a new model with random weights (no pretrained weights, ever)
  -> trains until 60 min (or 15 min for pilots) of counted time; the clock pauses
     for validation checks and checkpoints; torch.compile warm-up IS counted
  -> scores the final model on the full validation split (bits per byte)
        |
        v
results/runs/E01-N26_lr2e-3__s0.json   +   one row in results/compute_ledger.csv
```

If Modal loses the machine, it restarts the job automatically and the run resumes from its last checkpoint with its
counted time kept (tested on Modal, 1 Nov).

---

## Repository layout

Every folder has a README that explains each file.

| Folder / file | What it contains |
|---|---|
| [`DECLARATIONS.md`](DECLARATIONS.md) | The fixed rules, plus an Amendments table for every change |
| [`modal_app.py`](modal_app.py) | **The entry point**: launches runs on Modal (`launch`), recovers results (`recover`), prepares data (`prepare_data`), checks the GPU (`gpu_check`) |
| [`code/`](code/) | Training, model, evaluation and data-preparation code, tests, and the results summary script |
| [`configs/`](configs/) | One YAML file per run, grouped by experiment (`E00/`, `E01/`, ...) |
| [`data/`](data/) | The tokenizer and the data manifest (hashes, sizes). The token files themselves live on Modal |
| [`results/`](results/) | One JSON per run and the compute ledger (every GPU job, its time and cost) |
| [`report/`](report/) | The LaTeX report and its figures (not started yet) |
| `requirements.txt` | Exact package versions installed in the Modal container (used for every run) |
| `requirements-dev.txt` | Packages for your laptop: `modal` and `pyyaml` |
| `.gitattributes` | Forces LF line endings, so hashed files (e.g. `data/tokenizer.json`) match `data/manifest.json` on Windows |

Experiment numbering (used in config folders, run IDs and result files):

| ID | Experiment | Phase |
|---|---|---|
| E00 | Pipeline checks, throughput benchmarks, Modal resume test | 0 |
| E01 | Learning rate per model size (15-min pilots) and 60-min horizon check | 1 |
| E02 | Model size vs training time, depth vs width, noise calibration | 2 |
| E03 / E04 | Data quality / repetition | 3 |
| E05+ | Efficiency and optimiser experiments, numbered as they start | 4-5 |
| E99 | Final test-set evaluation | 6 |

---

## Team

Group 24. Names and student numbers: `authors.md` (to be added before submission).
