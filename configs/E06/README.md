# E06: shallow-and-wide vs deep-and-narrow at the same size (Phase 4b)

Two shapes with about N49's 49M parameters: 6 layers x 768 (wide) and 19 layers x 448 (deep), compared with N49's
11 x 576. Each shape got the full E01 tuning protocol (4 rates, 15-minute pilots, close calls with 3 seeds); then only the
picked rate was run for 60 minutes. There is one 60-minute file per learning rate so that no file had to be edited after
the pilots; the unused ones were never launched.

**Outcome:** 6 x 768 is better than N49 (-0.018, one seed, via +16 % training speed; tied at equal steps), 19 x 448 is
worse (+0.021).

- Write-up (question, design, rules, results, surprises): [docs/experiments/E06_depth_width.md](../../docs/experiments/E06_depth_width.md)
- Results: `results/runs/E06-*.json` ([results/runs/](../../results/runs/)); table with the declared checks: `python code/summarize.py E06`
- 16 configs, 10 run, 6 never run (options declared in advance that the rules did not call for; kept as the record of what was planned)

"BPB" is the final FineWeb-Edu validation bits per byte (mean over the seeds run). Minutes = counted training time.

| Config | Minutes | Seeds run | BPB | Role |
|---|---|---|---|---|
| `D19x448_lr1e-3_15min.yaml` | 15 | 0 | 1.4151 | 15-min LR pilot |
| `D19x448_lr1e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `D19x448_lr2e-3_15min.yaml` | 15 | 0, 1, 2 | 1.3846 (3 seeds) | 15-min LR pilot |
| `D19x448_lr2e-3_60min.yaml` | 60 | 0 | 1.2158 | **The pick**: 60-min run |
| `D19x448_lr4e-3_15min.yaml` | 15 | 0, 1, 2 | 1.3933 (3 seeds) | 15-min LR pilot |
| `D19x448_lr4e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `D19x448_lr5e-4_15min.yaml` | 15 | 0 | 1.4861 | 15-min LR pilot |
| `D19x448_lr5e-4_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `W6x768_lr1e-3_15min.yaml` | 15 | 0, 1, 2 | 1.3259 (3 seeds) | 15-min LR pilot |
| `W6x768_lr1e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `W6x768_lr2e-3_15min.yaml` | 15 | 0, 1, 2 | 1.3178 (3 seeds) | 15-min LR pilot |
| `W6x768_lr2e-3_60min.yaml` | 60 | 0 | 1.1768 | **The pick**: 60-min run |
| `W6x768_lr4e-3_15min.yaml` | 15 | 0 | 1.3602 | 15-min LR pilot |
| `W6x768_lr4e-3_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
| `W6x768_lr5e-4_15min.yaml` | 15 | 0 | 1.3702 | 15-min LR pilot |
| `W6x768_lr5e-4_60min.yaml` | 60 | **not run** |  | Unused: the pilots picked 2e-3 |
