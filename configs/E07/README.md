# E07: vocabulary size (Phase 5)

N49 (AdamW) trained with a 4,096, 16,384 or 32,768-token vocabulary instead of 8,192 (`data_variant`; the data is built by
[`tools/`](../../tools/) and stored per variant on Modal). Same tuning protocol as N49 had in E01. The `2.5e-4` and `8e-3`
pilots are edge extensions, launched only if a vocabulary's best rate had been at the edge of the grid (none was). One
60-minute file per learning rate; only the picked rate (2e-3 for all three) was run.

**Outcome:** 16k (3 seeds, -0.016) and 32k (1 seed, -0.017) are both better than 8k and indistinguishable from each other;
4k is a single-seed close call (+0.009, probably worse). About a third of each gap comes from the scoring window holding
more text at larger vocabularies.

- Write-up (question, design, rules, results, surprises): [docs/experiments/E07_vocabulary.md](../../docs/experiments/E07_vocabulary.md)
- Results: `results/runs/E07-*.json` ([results/runs/](../../results/runs/)); table with the declared checks: `python code/summarize.py E07`
- 36 configs, 15 run, 21 never run (options declared in advance that the rules did not call for; kept as the record of what was planned)

"BPB" is the final FineWeb-Edu validation bits per byte (mean over the seeds run). Minutes = counted training time.

| Config | Minutes | Seeds run | BPB | Role |
|---|---|---|---|---|
| `V16384_lr1e-3_15min.yaml` | 15 | 0 | 1.3658 | 15-min LR pilot |
| `V16384_lr1e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V16384_lr2.5e-4_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `V16384_lr2.5e-4_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V16384_lr2e-3_15min.yaml` | 15 | 0 | 1.3426 | 15-min LR pilot |
| `V16384_lr2e-3_60min.yaml` | 60 | 0, 1, 2 | 1.1781 (3 seeds) | **The pick**: 60-min run |
| `V16384_lr4e-3_15min.yaml` | 15 | 0 | 1.3656 | 15-min LR pilot |
| `V16384_lr4e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V16384_lr5e-4_15min.yaml` | 15 | 0 | 1.4252 | 15-min LR pilot |
| `V16384_lr5e-4_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V16384_lr8e-3_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `V16384_lr8e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V32768_lr1e-3_15min.yaml` | 15 | 0 | 1.3874 | 15-min LR pilot |
| `V32768_lr1e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V32768_lr2.5e-4_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `V32768_lr2.5e-4_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V32768_lr2e-3_15min.yaml` | 15 | 0 | 1.3498 | 15-min LR pilot |
| `V32768_lr2e-3_60min.yaml` | 60 | 0 | 1.1774 | **The pick**: 60-min run |
| `V32768_lr4e-3_15min.yaml` | 15 | 0 | 1.3996 | 15-min LR pilot |
| `V32768_lr4e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V32768_lr5e-4_15min.yaml` | 15 | 0 | 1.4746 | 15-min LR pilot |
| `V32768_lr5e-4_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V32768_lr8e-3_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `V32768_lr8e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V4096_lr1e-3_15min.yaml` | 15 | 0 | 1.3772 | 15-min LR pilot |
| `V4096_lr1e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V4096_lr2.5e-4_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `V4096_lr2.5e-4_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V4096_lr2e-3_15min.yaml` | 15 | 0 | 1.3548 | 15-min LR pilot |
| `V4096_lr2e-3_60min.yaml` | 60 | 0 | 1.2032 | **The pick**: 60-min run |
| `V4096_lr4e-3_15min.yaml` | 15 | 0 | 1.3841 | 15-min LR pilot |
| `V4096_lr4e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V4096_lr5e-4_15min.yaml` | 15 | 0 | 1.4434 | 15-min LR pilot |
| `V4096_lr5e-4_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `V4096_lr8e-3_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `V4096_lr8e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
