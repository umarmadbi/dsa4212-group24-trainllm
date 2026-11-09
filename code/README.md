# code/

All Python code for the project. Most of it runs **inside the Modal container**: you start it through
[`../modal_app.py`](../modal_app.py) from the repository root, not by calling these files directly. `summarize.py`, the
two fitting scripts and the tests run on your own computer. All scripts are run from the repository root.

The files stay in one flat folder on purpose: they import each other by file name, `modal_app.py` copies this folder into
the container as it is, every result file records the hash of the code that produced it, and the config files quote the
command `python code/summarize.py`. Moving files would make the repository differ from the code that ran.

## Training and the metric (run on Modal, for every training run)

| File | What it does |
|---|---|
| `train.py` | **The training loop used by every run.** Loads the tokens into memory, builds a fresh random model, trains it under the time budget (counted-time clock, time-based learning-rate schedule), saves checkpoints and resumes after a preemption, scores the final model and writes the result file. Also stops a run whose loss becomes NaN, and logs gradient norms and GPU clock, temperature and power. Data loaders for unique data, repeated subsets (E04), shifted windows (E04) and corpus mixtures (E03) |
| `model.py` | The GPT model, copied from nanoGPT (commit `3adf61e`) with three documented changes: no code path to load pretrained GPT-2 weights; logits for every position (needed for bits per byte); parameter and FLOP counting. Muon support is in `train.py` (PyTorch's `torch.optim.Muon` on the hidden weight matrices) |
| `bpb.py` | The bits-per-byte evaluator. Cuts the text into non-overlapping 512-token windows, scores every token once, divides the total loss by the exact number of bytes. The same code scores validation and test. `evaluate_bpb_sliding` (overlapping windows) is a diagnostic only, never the declared metric |
| `config.py` | Every run setting with its default (`Config`) and a strict YAML loader that rejects unknown fields and wrong types. Does not import torch, so `launch` checks configs on your laptop before any GPU starts. Field list: [configs/README.md](../configs/README.md) |

## Data preparation (run once on Modal CPU)

| File | What it does |
|---|---|
| `prepare_data.py` | Downloads the four corpora at pinned versions, splits documents into train / validation / test, removes duplicates across splits and corpora, trains the 8,192-token BPE tokenizer, converts all text to token files and records sizes and hashes in `data/manifest.json`. Run through `modal_app.py::rebuild_data`. The vocabulary variants of E07 are built by [`tools/`](../tools/) |

## Evaluation after training (run on Modal)

| File | What it does |
|---|---|
| `contamination.py` | 13-gram contamination check: looks up every 13-token passage of every validation and test document in the full train split of each training corpus and lists documents with at least 50 % of their passages there. Output: `results/contamination.json`. Run with `modal run --detach modal_app.py::contamination` |
| `rescore.py` | Re-scores a saved final model on any validation set, optionally without the contaminated documents, and checks it reproduces the stored score exactly. `--stride N` adds the overlapping-window score. Output: `results/rescore/`. Run with `modal run modal_app.py::rescore --run-ids ...` from the workspace that trained the model |
| `evaluate_test.py` | The test-set lock and the one-time test scoring. Refuses any run not listed in `configs/frozen_configs.yaml` and refuses while that list is not `frozen`. Scores a run on the test split of its own vocabulary, also without contaminated test documents and (core models) with overlapping windows; writes a few text samples for marked runs. Called only through `modal run --detach modal_app.py::test_eval`, which also refuses uncommitted code and any run already scored. Output: `results/test/` |

## Results summaries and post-hoc analysis (run on your computer)

| File | What it does |
|---|---|
| `summarize.py` | **Prints the results table of one experiment** from `results/runs/`: `python code/summarize.py E05`. For each experiment it applies the decision rules declared before its runs and prints the checks beside them (slow machines, compile warm-up, grid edges, seed noise, speed-adjusted and equal-text views). `TEST` prints the test scores next to validation. Standard library only |
| `fit_scaling_law.py` | Diagnostic (D24): fits a Chinchilla-style loss law to our unique-data runs and finds the equal-FLOP optimum. Never used for a decision. Needs numpy and scipy |
| `fit_repetition.py` | Diagnostic (D25): converts each repeated-data run (E04) into "how many tokens of new data it was worth" with the loss law above, and fits Muennighoff et al.'s repetition half-life R*. Never used for a decision. Needs numpy and scipy |

## Subfolders

| Folder | Contents |
|---|---|
| [`tests/`](tests/) | 22 correctness tests that run on CPU in about 2 minutes. Run after any change to this folder |
| [`plots/`](plots/) | One script per report figure, reading only from `results/` |

## Running `train.py` without Modal (debugging only)

```powershell
python code/train.py --config configs/E01/N7_lr2e-3.yaml --seed 0 --data_dir <folder with tokens/ and tokenizer/> --out_root runs_local --set budget_s=60 compile=false
```

`--set key=value` overrides any config field. Results from a laptop or Colab are never reported: speed differs between
GPUs, and the budget is defined on the L4.

## Where the declared rules live in the code

| Rule ([DECLARATIONS.md](../docs/DECLARATIONS.md)) | Where |
|---|---|
| Clock starts before step 1, GPU synchronised, paused for evaluation and checkpoints, stops at the first step after the budget | `train.py`: `BudgetClock`, main loop |
| Cosine decay driven by the fraction of the budget used, ending exactly at the cap | `train.py`: `lr_at` |
| Each training window seen once per pass, passes counted exactly (repetition experiment) | `train.py`: `WindowLoader` |
| Checkpoint holds model, optimiser, counted time, tokens, data position and random states; resuming never resets the budget | `train.py`: `save_ckpt`, resume block |
| The final model is evaluated (no best-checkpoint selection) | `train.py`: end of `run` |
| No pretrained weights | `model.py` (`from_pretrained` removed; a test checks it) |
| Every run launches from committed code and records its commit and code hash | `modal_app.py`: `launch`, `_git_info` |
| Test split only for frozen runs, scored once | `evaluate_test.py`, `modal_app.py`: `test_eval` |
