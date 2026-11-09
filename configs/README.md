# configs/

**One YAML file = one training run.** A file holds every setting for that run; settings left out take the default in
[`code/config.py`](../code/config.py). Each experiment has its own folder with a README that lists every config, which
seeds were run, the result, and which configs were never run and why.

```powershell
# from the repository root, after committing (launch refuses uncommitted code)
modal run modal_app.py::launch --configs configs/E02/N49.yaml --seeds 0,1,2 --kind final
```

`launch` checks every config on your laptop before any GPU starts, so a typo fails immediately and costs nothing.
Do not edit a config that has already been run: each result file stores the hash of its config.

## Experiments

| Folder | Experiment | Phase |
|---|---|---|
| [`E00/`](E00/) | Pipeline checks, speed benchmarks, Modal resume test (preparation, never reported) | 0 |
| [`E01/`](E01/) | Learning rate per model size: 15-min pilots and a 60-min horizon check (preparation) | 1 |
| [`E02/`](E02/) | Which model size is best in 60 minutes; seed-noise calibration; N49 becomes the baseline | 2 |
| [`E03/`](E03/) | Data quality: FineWeb-Edu vs FineWeb vs DCLM, and a FineWeb-Edu/DCLM mixture | 3a |
| [`E04/`](E04/) | Repeated data: 2-32 passes over a small slice, plus two follow-ups | 3b |
| [`E05/`](E05/) | Muon vs AdamW | 4a |
| [`E06/`](E06/) | Shallow-and-wide vs deep-and-narrow at the same size | 4b |
| [`E07/`](E07/) | Vocabulary size: 4k, 16k, 32k vs 8k | 5 |
| [`E08/`](E08/) | The combined recipes RA and RB, 3 seeds each | 6 |
| [`frozen_configs.yaml`](frozen_configs.yaml) | The test-set lock (see below) | 6 |

## Naming

The result of `configs/E02/N49.yaml` with seed 0 is `results/runs/E02-N49__s0.json`: the config's `run_id` plus
`__s<seed>`. Model sizes are named by total parameters in millions; every model has a 512-token context and heads of 64
dimensions:

| Name | Layers x width | Parameters | Heads |
|---|---|---|---|
| N7 | 6 x 256 | 7.0M | 4 |
| N11 | 7 x 320 | 11.4M | 5 |
| N17 | 8 x 384 | 17.5M | 6 |
| N26 | 9 x 448 | 25.6M | 7 |
| N36 | 10 x 512 | 35.9M | 8 |
| N49 | 11 x 576 | 48.8M | 9 |
| N65 | 12 x 640 | 64.6M | 10 |

Other parts of a name: `lr2e-3` = peak learning rate (`muon_lr` for Muon runs); `15min` / `60min` = counted training
time; `W6x768` / `D19x448` = a model shape; `V16384` = vocabulary size; `muon` = Muon optimiser; `repK` = data repeated
about K times. E00 used older names: S1 = N7, S2 = N17, S3 = N36, S4 = N65, S5 = 91.6M, S6 = 123.6M.

## frozen_configs.yaml: the test-set lock

Not a run config. It lists every trained model whose test score the report may quote (57 runs), with its tier and the
Modal workspace that holds the trained model. `code/evaluate_test.py` scores a model on the test split only if it is
listed there and the file's `status` is `frozen`. It was set to `frozen` on 6 Nov 2026 after the last decision (D45) and
committed before any test score was computed; its git history is the evidence that the test set did not influence any
decision. **Never edit it.**

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
