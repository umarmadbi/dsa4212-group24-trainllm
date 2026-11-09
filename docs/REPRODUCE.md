# How to reproduce the results

Step-by-step instructions to rebuild the data, rerun any experiment, recheck the evaluation, and trace each number in the
report to the configs and result files behind it. All commands run from the repository root. Commands are written for
Windows PowerShell; they work the same on macOS/Linux.

## 1. Environment

| Where | What | Pinned in |
|---|---|---|
| Your computer | Launches jobs, summarises results. **Python 3.12 or newer** (the pinned `scipy` does not install on 3.11) with `modal`, `pyyaml`, `scipy`, and git | [`setup/requirements-dev.txt`](../setup/requirements-dev.txt) |
| Modal container | Runs every training, data and evaluation job: Debian slim, Python 3.12, `gcc`/`g++` (for `torch.compile`), PyTorch 2.15.0, one NVIDIA L4 for GPU jobs | [`requirements.txt`](../requirements.txt), image definition at the top of [`modal_app.py`](../modal_app.py) |

```powershell
git clone https://github.com/umarmadbi/dsa4212-group24-trainllm.git
cd dsa4212-group24-trainllm
py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1      # macOS/Linux: python3.12 -m venv .venv; source .venv/bin/activate
pip install -r setup/requirements-dev.txt
modal setup                                   # log in; Modal account with credit
modal run modal_app.py::gpu_check             # must report an NVIDIA L4 with bf16
```

No Modal needed: the CPU correctness tests (`pip install torch numpy pyyaml tokenizers`; the first install of torch takes a few minutes):

```powershell
python code/tests/test_pipeline.py            # expected last line: 22/22 passed (2-5 min on a laptop CPU)
python tools/test_prepare_vocab.py            # the vocabulary builder
```

## 2. Data

The token files (~15 GB) are not in the repository; each Modal workspace builds its own copy in a Volume named
`small-llm-data`, from the dataset versions pinned in [`data/manifest.json`](../data/manifest.json).

```powershell
modal run --detach modal_app.py::rebuild_data     # CPU only, ~1 hour, ~$0.60
modal run modal_app.py::verify_data               # afterwards: must print DATA CHECK PASSED
```

`verify_data` re-hashes every token file and the tokenizer against the manifest. The vocabulary variants (needed for E07
and E08) are built the same way:

```powershell
modal run --detach tools/vocab_app.py::build      # 4k, 16k, 32k (+ a check that 8k reproduces Phase 0), ~30-60 min CPU
modal run tools/vocab_app.py::fetch               # must print "8,192 REPRODUCTION CHECK: PASSED"
modal run tools/vocab_app.py::verify              # must print "VOCAB DATA CHECK PASSED"
```

## 3. Training runs

One config file is one run; seeds are given at launch. Each (config, seed) pair runs in its own container on its own L4.

```powershell
modal run modal_app.py::launch --configs configs/E05/N49_muon_lr2e-3_60min.yaml --seeds 0,1,2 --kind final
modal run --detach modal_app.py::launch --configs "configs/E07/V16384_lr2e-3_15min.yaml,configs/E07/V16384_lr4e-3_15min.yaml" --seeds 0 --kind pilot
```

- `launch` refuses to start from uncommitted code (commit first); each result records the commit and code hash.
- `--kind` is a label for the ledger (`pilot` for 15-minute tuning runs, `final` for 60-minute runs).
- `--member <name>` records who launched it (default `umarmadbi`).
- When a run finishes, `results/runs/<run_id>__s<seed>.json` and a row in `results/compute_ledger.csv` are written.
- If the laptop loses its connection (or with `--detach`), the run continues on Modal; fetch it later with
  `modal run modal_app.py::recover --run-ids "<run_id>__s<seed>" --configs <config> --seeds <seed> --kind <kind>`.
- Cost: a 60-minute run takes about 62 minutes of container time (~$0.83); a 15-minute pilot ~$0.22.
- Results vary between machines by about 0.006 BPB for a single run (cloud GPUs differ in speed); this is why the
  decision rules use 3 seeds for close calls.

## 4. Which configs and commands produce each result

`python code/summarize.py <E..>` prints each experiment's table and applies the decision rules that were declared before
its runs. Seeds below are the ones that were actually run (also listed per config in each `configs/E0x/README.md`).

| Result | Configs (seeds) | Summary | Write-up |
|---|---|---|---|
| Speed of each size; compile on | `configs/E00/bench_*.yaml` (0) | `summarize.py E00` | [E00](experiments/E00_pipeline_and_benchmarks.md) |
| Learning rate per size | `configs/E01/N*_lr*.yaml` (0; close calls 0-2); horizon check `N26_lr{2e-3,4e-3}_60min.yaml` (0) | `summarize.py E01` | [E01](experiments/E01_learning_rate_tuning.md) |
| Best size in 60 min; seed noise | `configs/E02/N{7,11,17}.yaml` (0), `N{36,49,65}.yaml` (0-2), `N26.yaml` (1-4) + `E01/N26_lr4e-3_60min` (0) | `summarize.py E02`; fit: `python code/fit_scaling_law.py` | [E02](experiments/E02_model_size.md) |
| Data quality | `configs/E03/N49_{fineweb,dclm,mix_fwe60_dclm40}.yaml` (0-2) vs `E02/N49.yaml` (0-2); LR checks `*_15min` (0) | `summarize.py E03` | [E03](experiments/E03_data_quality.md) |
| Repetition | `configs/E04/N49_rep{2,4,8}.yaml` (0-2), `rep{16,32}`, `N{17,26,65}_U36M`, `rep8_shift` (0) | `summarize.py E04`; fit: `python code/fit_repetition.py` | [E04](experiments/E04_repetition.md) |
| Muon vs AdamW | `configs/E05/N49_muon_lr*_15min.yaml` (0; 2e-3, 4e-3: 0-2); `N49_muon_lr2e-3_60min.yaml` (0-2); `lr{1e-3,4e-3}_60min` (0) | `summarize.py E05` | [E05](experiments/E05_muon.md) |
| Depth vs width | `configs/E06/{W6x768,D19x448}_lr*_15min.yaml` (0; close calls 0-2); `*_lr2e-3_60min.yaml` (0) | `summarize.py E06` | [E06](experiments/E06_depth_width.md) |
| Vocabulary size | `configs/E07/V*_lr*_15min.yaml` (0, 4 rates); `V{4096,32768}_lr2e-3_60min` (0), `V16384_lr2e-3_60min` (0-2) | `summarize.py E07` | [E07](experiments/E07_vocabulary.md) |
| Combined recipes | `configs/E08/*_muon_lr*_15min.yaml` (0; close calls 0-2); `W6x768_V16384_muon_lr4e-3_60min.yaml`, `N49_V16384_muon_lr2e-3_60min.yaml` (0-2) | `summarize.py E08` | [E08](experiments/E08_final_recipe.md) |
| Test scores | the 57 runs in `configs/frozen_configs.yaml` | `summarize.py TEST` | [E08](experiments/E08_final_recipe.md), D47 in [decisions.md](decisions.md) |

## 5. Evaluation checks after training

These need the trained models, which are stored in the `small-llm-runs` Volume of the workspace that trained them (the
workspace of each run is in the ledger and in `configs/frozen_configs.yaml`).

```powershell
modal run --detach modal_app.py::contamination           # 13-gram check -> results/contamination.json (CPU, ~10-20 min)
modal run modal_app.py::rescore --run-ids "E03-N49_dclm__s0" --exclude contamination    # -> results/rescore/<run>__nocontam.json
modal run modal_app.py::rescore --run-ids "E07-V4096_lr2e-3_60min__s0" --sets fineweb_edu --stride 128   # overlapping windows
```

Each re-score first reproduces the stored validation score exactly, then adds the new score.

**Test set.** `modal run --detach modal_app.py::test_eval` scores the frozen runs of the current workspace on the test
split. It refuses unless `configs/frozen_configs.yaml` has status `frozen` and is committed, the code is committed, and
the run has never been scored before (a file in `results/test/` or on the Volume). Our test scores were computed once,
on 6 Nov 2026, and are in `results/test/`. Someone retraining the models in their own workspace would have to move our
`results/test/` files aside first, since the lock is meant to stop any model from being scored twice.

## 6. Checking that the results reproduce

Four levels, from free to a few dollars. Each one checks something different.

| Check | Cost | What it shows | Pass criterion |
|---|---|---|---|
| `python code/tests/test_pipeline.py` | free, CPU | The training loop, metric, data loaders, Muon, resume and test lock behave correctly (see [code/tests/README.md](../code/tests/README.md)) | `22/22 passed` |
| `python code/summarize.py <E00..E08 / TEST>` | free | Every table in the write-ups is rebuilt from the saved result files, applying the declared rules; nothing is typed by hand | Same numbers as the write-ups |
| `modal run modal_app.py::verify_data` (after `rebuild_data`) | cents, CPU | Your data is byte-identical to ours: every token file and the tokenizer match the SHA-256 hashes in `data/manifest.json` | `DATA CHECK PASSED` |
| Retrain a run and compare (below) | a few cents to $0.83 per run | The method reproduces our numbers on a new account | Within the stated range |

Runs to retrain, cheapest first. Results are not bit-identical on cloud GPUs: the budget is time, so a faster or slower
L4 reads a few percent more or less text, which moved single runs by up to ~0.006 BPB (seed noise measured in E02).

| Config (seed 0) | Time / cost | Our result(s), BPB | Expect |
|---|---|---|---|
| `configs/E00/resume_test_N7.yaml` (`--kind smoke`) | 4 min, a few cents | 1.4675; the job kills itself once and resumes | `attempts` = 2 in the result file; counted time 240 s |
| `configs/E01/N49_lr2e-3.yaml` (`--kind pilot`) | 15 min, ~$0.22 | 1.3577 / 1.3439 / 1.3507 (seeds 0-2) | ~1.34-1.36 |
| `configs/E02/N49.yaml` (the baseline) | 60 min, ~$0.83 | 1.1936 / 1.1961 / 1.1937 | ~1.19-1.20 |
| `configs/E05/N49_muon_lr2e-3_60min.yaml` (Muon) | 60 min, ~$0.83 | 1.1686 / 1.1679 / 1.1684 | ~1.16-1.18 |
| `configs/E08/N49_V16384_muon_lr2e-3_60min.yaml` (final recipe; needs the 16k data, section 2) | 60 min, ~$0.83 | 1.1537 / 1.1554 / 1.1502 | ~1.145-1.165 |

A new result is written to `results/runs/<run_id>__s<seed>.json`; our original is moved to `results/runs/_replaced/`
(never deleted) and is still in git. Compare with `python code/summarize.py <experiment>` or `git diff results/runs/`, and
restore our files afterwards with `git checkout -- results` (and delete any new files in `results/runs/_replaced/`).

For the trained models we still hold, `rescore` (section 5) reproduces each stored validation score exactly
(difference 0.0) before computing anything new; every re-score in `results/rescore/` passed this check.

## 7. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled on this system` | Windows blocks scripts by default. Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `cmd` with `.venv\Scripts\activate.bat` |
| `pip install` fails with "requirements are unsatisfiable" or "No matching distribution found for scipy==1.18.1" | Python older than 3.12. Check with `python --version`; delete `.venv` and recreate it with `py -3.12 -m venv .venv` (install Python 3.12 from python.org if `py -3.12` is not found) |
| `'modal' is not recognized` / `command not found: modal` | The virtual environment is not active in this terminal. Activate it again (step 1) |
| `Token missing. Could not authenticate client` | Not logged in to Modal: run `modal setup` (or `modal token new`). `modal profile current` shows which account is active |
| Jobs bill the wrong Modal account | `modal profile list`, then `modal profile activate <name>`, or for one terminal `$env:MODAL_PROFILE = "<name>"` (macOS/Linux: `export MODAL_PROFILE=<name>`) |
| `REFUSING TO LAUNCH: code/, configs/, modal_app.py or requirements.txt differ from the last commit` | By design: every run must come from committed code. Run `git status`, then commit (or `git stash`) your changes. `--allow-dirty` exists for throwaway tests only; such results are marked `+dirty` |
| `--configs "configs/E07/*.yaml"` fails on Windows | List the files separated by commas instead: `--configs "configs/E07/a.yaml,configs/E07/b.yaml"` |
| The run stays queued for a long time | Modal is waiting for a free L4 (we saw waits from minutes to over an hour). The budget clock only starts once training starts; queueing costs nothing. Launch with `--detach` and check the Modal dashboard |
| The terminal lost its connection / the laptop went to sleep | The job keeps running on Modal (always with `--detach`). Fetch it later with `modal run modal_app.py::recover --run-ids "<run_id>__s<seed>" --configs <config> --seeds <seed> --kind <kind>`. The ledger may show an extra `failed` row for the dropped connection |
| `DATA CHECK FAILED` or a run cannot find `/data/tokens/...` | The data build has not finished, or was interrupted. Wait for `rebuild_data` to finish (Modal dashboard -> Apps), then run `verify_data` again; rerun `rebuild_data` if files are missing |
| A vocabulary run (E07/E08) cannot find `/data/vocab/v16384/` | The variant data is built separately: `modal run --detach tools/vocab_app.py::build`, then `fetch` and `verify` (section 2) |
| `PermissionError` when the result is written to `results/compute_ledger.csv` (Windows) | The ledger is open in Excel or another program. Close it and run `recover` for the run; the result is safe on Modal |
| `test lock: already scored` from `test_eval` | By design: test scores are computed once. See section 5 |
| `python code/fit_scaling_law.py` reports `No module named 'scipy'` | Install the laptop packages: `pip install -r setup/requirements-dev.txt` |
| `recover` says there is no result for a run that finished | The terminal is pointing at a different Modal account from the one that ran the job. `modal profile current`, then switch to the right one (see "Jobs bill the wrong Modal account") and run `recover` again |
| `rescore`, `contamination` or `test_eval` lost its connection before printing the result | The job's output is saved on Modal first. Rerun the same command with `--fetch-only` (from the same account) to collect it without recomputing |
| `summarize.py` finds no runs, or a script cannot find `results/` | Scripts are run from the repository root (`python code/summarize.py E02`), not from inside `code/` |
| Tests end with `21/22 passed` and `No module named 'tokenizers'` | `pip install tokenizers`, then run the tests again |

## 8. Figures

Each report figure is made by one script in [`code/plots/`](../code/plots/) that reads only from `results/` and writes to
`report/figures/` (see the table in that folder's README).
