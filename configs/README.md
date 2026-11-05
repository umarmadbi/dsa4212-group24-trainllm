# configs/

**One YAML file = one run.** A file holds every setting for that run. Folders are named by experiment number
(`E00/`, `E01/`, ...). The same config can be run with several seeds (`--seeds 0,1,2`); each seed is its own run.

```powershell
# from the repo root, after committing
modal run modal_app.py::launch --configs configs/E01/N26_lr2e-3.yaml --seeds 0 --kind pilot
```

`launch` checks every config on your laptop before any GPU starts, so a typo fails immediately and costs nothing.

## Naming

The result of `configs/E01/N26_lr2e-3.yaml` with seed 0 is `results/runs/E01-N26_lr2e-3__s0.json`: the config's
`run_id` plus `__s<seed>`. Model sizes are named by total parameters in millions:

| Name | Layers x width | Parameters | Heads (64 dims each) |
|---|---|---|---|
| N7 | 6 x 256 | 7.0M | 4 |
| N11 | 7 x 320 | 11.4M | 5 |
| N17 | 8 x 384 | 17.5M | 6 |
| N26 | 9 x 448 | 25.6M | 7 |
| N36 | 10 x 512 | 35.9M | 8 |
| N65 | 12 x 640 | 64.6M | 10 |

(E00 used older names for some of the same sizes: S1 = N7, S2 = N17, S3 = N36, S4 = N65; S5 and S6, 91.6M and
123.6M, were dropped from the size grid.)

## E00/: pipeline checks (Phase 0, preparation only, never reported as results)

| File | What it is |
|---|---|
| `bench_S1.yaml` ... `bench_S6.yaml` | 5-minute speed benchmarks of 6 model sizes on the L4 (tokens per second, memory) |
| `bench_S3_nocompile.yaml` | S3 without `torch.compile`, to decide whether compile is worth its warm-up time (it is: +20 % speed) |
| `bench_S1_micro64.yaml` | S1 with a bigger micro-batch, to see if it is faster (it is not) |
| `resume_test_N7.yaml` | 4-minute run that kills itself once at step 1500, to prove Modal restarts it and it resumes with its counted time kept (passed) |

## E01/: learning rate per model size (Phase 1, preparation only)

| Files | What they are |
|---|---|
| `N<size>_lr<rate>.yaml` | 15-minute pilots: 4 learning rates per size, factor 2 apart (N7/N11: 1e-3 to 8e-3; others: 5e-4 to 4e-3). The best per size is used in Phase 2 |
| `N36_lr8e-3.yaml`, `N11_lr1.6e-2.yaml` | Extra rates beyond the edge of the grid, because that size was best at the edge (declared rule) |
| `N49_lr{5e-4,1e-3,2e-3,4e-3}.yaml` | LR pilots for the extra size N49, run only if rule P2b of E02 fires |
| `N26_lr{1e-3,2e-3,4e-3}_60min.yaml` | **Horizon check**: the same N26 runs for the full 60 minutes, to check a rate chosen in 15 minutes still wins over an hour |

## E02/: which model size is best in 60 minutes (Phase 2, final runs)

| Files | What they are |
|---|---|
| `N7.yaml` ... `N65.yaml` | The 6 sizes, each for the full 60 minutes at its E01 learning rate (N7, N11, N65: 2e-3; N17, N26, N36: 4e-3). Otherwise identical |
| `N26.yaml` seeds 1-4 | Noise calibration: the same run 4 more times with different seeds (seed 0 is the E01 60-min run, counted here) |
| `N4.yaml`, `N92.yaml` | Extra sizes beyond the edges, run only if the smallest or largest size wins (their LR is not tuned) |
| `N49.yaml` | Extra size (11 x 576, 48.8M) between N36 and N65, run only if the fitted optimum falls in that gap (rule P2b). Its LR is tuned first with `configs/E01/N49_lr*.yaml`; the LR in the file is a placeholder until then |

Every E02 config scores the final model on all four validation splits (`eval_corpora`): FineWeb-Edu is the headline
number used for decisions; FineWeb, DCLM and Wikipedia are extra information, scored after training (outside the budget).

## E03/: data quality (Phase 3a, final runs)

| Files | What they are |
|---|---|
| `N49_fineweb.yaml`, `N49_dclm.yaml` | The baseline size N49 for 60 minutes on raw FineWeb or on DCLM instead of FineWeb-Edu (seeds 0-2). The FineWeb-Edu arm is `configs/E02/N49.yaml` (its runs are reused). Judged on the average over all four validation sets |
| `N49_{fineweb,dclm}_lr{2e-3,4e-3}_15min.yaml` | Learning-rate check (rule R6, D25): 15-minute pilots, seed 0. N49's learning rate was tuned on FineWeb-Edu only; these check that 2e-3 is not clearly worse than 4e-3 on the other two corpora. Preliminary work (`--kind pilot`), not part of the arms |
| `N49_mix_fwe60_dclm40.yaml` | Follow-up R8 (D26): N49 trained on a 60/40 mix of FineWeb-Edu and DCLM (by tokens), seeds 0-2. Each corpus draws from a 1B-token random pool. Judged like the other arms |

## E04/: repetition (Phase 3b, final runs)

| Files | What they are |
|---|---|
| `N49_rep2.yaml` ... `N49_rep32.yaml` | N49 for 60 minutes on a small random slice of FineWeb-Edu, sized so the hour reads it about 2, 4, 8, 16 or 32 times. Compared with N49 on unique data |
| `N17_U36M.yaml`, `N26_U36M.yaml`, `N65_U36M.yaml` | Follow-up F2 (D26): other model sizes on the same 36.25M-token subset as `N49_rep8.yaml`. Which size is best when unique data is limited? Seed 0 |
| `N49_rep8_shift.yaml` | Follow-up F3 (D26): as `N49_rep8.yaml`, but the window boundaries move every pass (`train_subset_shift`). Does showing identical samples every pass cause the damage? Seed 0 |

## E05/: Muon vs AdamW (Phase 4a)

| Files | What they are |
|---|---|
| `N49_muon_lr{1e-3,2e-3,4e-3,8e-3}_15min.yaml` | Muon learning-rate pilots at N49 (15 min, seed 0), the same protocol AdamW got in E01 |
| `N49_adamw_lr{1e-3,4e-3}_60min.yaml` | CONDITIONAL (D30): AdamW N49 for 60 min at its two neighbouring LRs, run only if Muon's horizon check is contradicted, so both optimisers get the same 60-min LR check |
| `N49_muon_lr{1e-3,2e-3,4e-3,8e-3}_60min.yaml` | N49 with Muon for 60 minutes, one file per LR (D30: no placeholder to edit). The pilot pick is run with seeds 0-2 (the final comparison with E02-N49, AdamW); the horizon-check LR(s) with seed 0. `python code/summarize.py E05` prints which to launch |

## E06/: depth vs width (Phase 4b)

| Files | What they are |
|---|---|
| `W6x768_lr*_15min.yaml`, `D19x448_lr*_15min.yaml` | Learning-rate pilots (5e-4, 1e-3, 2e-3, 4e-3; the declared E01 protocol, D30) for a shallow-wide (6 x 768) and a deep-narrow (19 x 448) model of N49's size |
| `W6x768_lr*_60min.yaml`, `D19x448_lr*_60min.yaml` | The two shapes for 60 minutes, one file per LR; only the picked LR is run (seed 0; seeds 1, 2 on a close call). Compared with N49 (11 x 576). `python code/summarize.py E06` prints which to launch |

## E07/: vocabulary size (Phase 5)

| Files | What they are |
|---|---|
| `V4096_lr*_15min.yaml`, `V16384_lr*_15min.yaml`, `V32768_lr*_15min.yaml` | Learning-rate pilots (5e-4, 1e-3, 2e-3, 4e-3: the grid N49 had in E01) for N49 (11 x 576, AdamW) with a 4,096 / 16,384 / 32,768-token vocabulary (`data_variant`, data built by `tools/prepare_vocab.py`). The `lr2.5e-4` and `lr8e-3` files are edge extensions, run only if `summarize.py E07` asks for them |
| `V*_lr*_60min.yaml` | The 60-minute runs, one file per LR; only the picked LR is run (seed 0; seeds 1, 2 on a close call). Compared with E02-N49 (8,192 vocabulary). `python code/summarize.py E07` prints which to launch. Rules: decision D36 |

## Settings (every field a config can have)

Fields left out take the default from `code/config.py`. Unknown fields are rejected.

| Field | Default | Meaning |
|---|---|---|
| `run_id` | | Name of the run (`E<NN>-<description>`); must be unique |
| `experiment` | `E00` | Experiment number, used to group results |
| `corpus` | `fineweb_edu` | Training text: `fineweb_edu`, `fineweb` or `dclm` |
| `train_tokens_limit` | 0 | If > 0, train only on the FIRST N tokens in file order (not a random sample; not used for results) |
| `train_subset_tokens` | 0 | If > 0, train on a random nested subset of about N tokens (whole windows, fixed key 1234): the repetition experiment E04 |
| `train_subset_shift` | false | With `train_subset_tokens`: move the window boundaries every pass (E04 F3, D26) |
| `corpus_mix` | [] | Mix corpora by token share, e.g. `["fineweb_edu=0.6", "dclm=0.4"]`; needs `corpus: mix` (E03 mixture, D26) |
| `mix_pool_tokens` | 1,000,000,000 | Size of each corpus's random pool in a mixture |
| `optimizer` | adamw | `adamw` (baseline) or `muon` (Muon on the hidden block matrices, AdamW for the rest; E05) |
| `muon_lr` | 0 | Peak Muon learning rate (needed for `muon`); the same warm-up/cosine schedule |
| `muon_momentum`, `muon_ns_steps`, `muon_adjust_lr` | 0.95, 5, match_rms_adamw | Muon settings (PyTorch defaults; update size matched to AdamW) |
| `eval_corpora` | `[fineweb_edu]` | Validation splits scored at the end; the first one is the headline number |
| `data_variant` | (empty) | Empty = the 8,192 baseline data. `v<V>` = the vocabulary variant at `/data/vocab/v<V>/` (Phase 5); must equal `vocab_size`, training corpus `fineweb_edu` only. With a variant the learning-curve subset is cut to the same text as the baseline's, and the data files are hashed into the result |
| `curve_corpus`, `curve_tokens` | `fineweb_edu`, 262,144 | Small validation subset scored during training for the learning curve |
| `n_layer`, `n_head`, `n_embd` | | Model shape (layers, attention heads, width) |
| `block_size` | 512 | Context length in tokens |
| `vocab_size` | 8192 | Must match the tokenizer |
| `dropout`, `bias` | 0.0, false | As in nanoGPT |
| `micro_batch`, `grad_accum` | | Sequences per forward pass and passes per optimiser step. We use 32 x 2 x 512 = **32,768 tokens per step** for every size |
| `lr` | | **Peak learning rate** (the thing E01 tunes) |
| `min_lr_ratio` | 0.1 | Final learning rate = 10 % of the peak |
| `warmup_steps` | 200 | Linear warm-up steps |
| `weight_decay`, `beta1`, `beta2`, `grad_clip` | 0.1, 0.9, 0.95, 1.0 | AdamW settings and gradient clipping |
| `budget_s` | 3600 | **Counted training time in seconds**: 3600 for real runs, 900 for 15-min pilots |
| `eval_every_frac` | 0.05 | Learning-curve point every 5 % of the budget (clock paused) |
| `ckpt_every_s` | 600 | Checkpoint every 10 min of counted time (clock paused) |
| `log_every_s` | 30 | Training log interval |
| `compile` | true | Use `torch.compile` (its warm-up counts toward the budget) |
| `dtype` | `bfloat16` | Precision |
| `eval_batch` | 32 | Windows per batch when scoring |
| `schedule`, `max_steps`, `ckpt_every_steps`, `test_crash_once_at_step` | | For tests only; leave them out |

## Adding a new experiment

1. Make a folder `configs/E<NN>/` and copy a similar config into it.
2. Change `run_id` (must start with `E<NN>-`) and `experiment`, then the settings you are testing.
3. Write numbers with a sign in the exponent or as plain decimals (`1.0e+9` or `1000000000`, not `1.0e9`).
   The loader converts them anyway, but this is the safe habit.
4. Commit, then launch.
