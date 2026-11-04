# code/

All Python code for the project. Most of it runs **inside the Modal container**: you launch it through
`modal_app.py` from the repo root, not by calling these files directly. The two exceptions are `summarize.py` and
the tests, which you run on your own computer.

| File | What it does | Who needs it |
|---|---|---|
| `train.py` | **The training loop used by every run.** Loads the tokens into memory, builds a fresh random model, trains it under the time budget (counted-time clock, time-based learning-rate schedule), saves checkpoints and resumes after a crash, scores the final model, and writes `result.json`. Also stops a run whose loss turns NaN, and logs gradient norms and GPU clock/temperature/power | Everyone (via `modal_app.py launch`) |
| `model.py` | The GPT model, copied from nanoGPT (commit `3adf61e`) with three documented changes: no code path to load pretrained GPT-2 weights; logits for every position (needed for bits per byte); parameter and FLOP counting | Read it to understand the model |
| `bpb.py` | The bits-per-byte evaluator. Cuts the text into non-overlapping 512-token windows, scores every token once, divides total loss by the exact number of bytes. The same code scores validation and (later) test | Read it to understand the metric |
| `config.py` | The list of every run setting with its default (`Config`), and a strict loader that rejects typos and wrong types in YAML files. Has no torch import, so `launch` can check configs on your laptop before any GPU starts | Anyone writing a config (field list in [configs/README.md](../configs/README.md)) |
| `prepare_data.py` | Phase 0 data preparation, run once on Modal CPU: downloads the four corpora at pinned versions, splits documents into train/validation/test, removes duplicates across splits and corpora, trains the 8,192-token BPE tokenizer, converts all text to token files, records hashes in `data/manifest.json` | Only to rebuild the data, or to copy it to a new Modal account |
| `summarize.py` | **Prints a results table for one experiment** from `results/runs/`. For E01 it also flags slow machines, warm-compile runs and grid-edge cases, shows the best learning rate per size, and gives the seed noise (sigma15). `E02`, `E03`, `E04` apply each experiment's declared decision rules (written in `docs/` before the runs) and print the checks beside them. Standard library only (no torch needed) | Everyone, after runs finish |
| `fit_scaling_law.py` | Post-hoc diagnostic (D24): fits a Chinchilla-style loss law to our own unique-data runs and finds the equal-FLOP optimum. Never used for a decision. Needs scipy | Report writers |
| `fit_repetition.py` | E04 diagnostic (D25): turns each repeated-data run into "how many tokens of new data it was worth" using the loss law above, and fits Muennighoff et al.'s repetition half-life R*. Never used for a decision. Needs scipy | Report writers |
| `contamination.py` | **13-gram contamination check** (D25): looks up every 13-token passage of every validation and test document in the full train split of each training corpus, and lists documents with >= 50 % of their passages there. Run on Modal: `modal run --detach modal_app.py::contamination` | Before interpreting E03; before Phase 6 |
| `rescore.py` | **Re-scores a saved final model** on any validation set, optionally without some documents (contamination sensitivity), and checks it reproduces the stored score. Run on Modal from the workspace that trained the model: `modal run modal_app.py::rescore --run-ids ...` | E03 R3; N26 seed 0's missing sets |
| `tests/` | Correctness tests for the pipeline (see [tests/README.md](tests/README.md)) | Anyone changing `train.py`, `model.py` or `bpb.py` |
| `plots/` | Scripts that make the report figures (see [plots/README.md](plots/README.md)) | Report writers |

## Quick example

```powershell
python code/summarize.py E01     # table of all E01 runs + best learning rate per size
python code/summarize.py E00     # the throughput benchmarks
```

## Running `train.py` without Modal (debugging only)

```powershell
python code/train.py --config configs/E01/N7_lr2e-3.yaml --seed 0 --data_dir <folder with tokens/ and tokenizer/> --out_root runs_local --set budget_s=60 compile=false
```

`--set key=value` overrides any config field. Results from your laptop or Colab are **never reported**: speed
differs between GPUs, and the budget is defined on the L4.

## Where the project rules live in the code

| Rule (DECLARATIONS.md) | Where |
|---|---|
| Clock starts before step 1, GPU synchronised, paused for eval/checkpoints, stop at the first step after the budget | `train.py`: `BudgetClock`, main loop |
| Cosine decay driven by the fraction of the budget used, ending exactly at the cap | `train.py`: `lr_at` |
| Each training window seen once per pass, passes counted exactly (repetition experiment) | `train.py`: `WindowLoader` |
| Checkpoint holds model, optimiser, counted time, tokens, data position, RNG; resume never resets the budget | `train.py`: `save_ckpt`, resume block |
| Final model evaluated (no best-checkpoint selection) | `train.py`: end of `run` |
| No pretrained weights | `model.py` (`from_pretrained` removed; a test checks it) |
