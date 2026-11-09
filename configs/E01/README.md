# E01: learning rate per model size (Phase 1)

Preparation for E02: a fair size comparison needs every size trained at a good learning rate. Each size got 4 peak
learning rates a factor 2 apart in 15-minute pilots; close calls were repeated with seeds 1 and 2; a size whose best rate
was at the edge of the grid got one extra rate. A 60-minute horizon check at N26 tested whether a rate picked in 15
minutes still wins over the full hour (it does).

**Outcome:** 2e-3 or 4e-3 for every size (N7, N11, N65: 2e-3; N17, N26, N36: 4e-3; N49: 2e-3). The two are close for
every size; 8e-3 breaks the larger models (N36 at 8e-3 reaches only 2.03).

- Write-up (question, design, rules, results, surprises): [docs/experiments/E01_learning_rate_tuning.md](../../docs/experiments/E01_learning_rate_tuning.md)
- Results: `results/runs/E01-*.json` ([results/runs/](../../results/runs/)); table with the declared checks: `python code/summarize.py E01`
- 33 configs, 32 run, 1 never run (options declared in advance that the rules did not call for; kept as the record of what was planned)

"BPB" is the final FineWeb-Edu validation bits per byte (mean over the seeds run). Minutes = counted training time.

| Config | Minutes | Seeds run | BPB | Role |
|---|---|---|---|---|
| `N11_lr1.6e-2.yaml` | 15 | 0 | 1.3704 | Edge extension (N11's best was at the top of its grid) |
| `N11_lr1e-3.yaml` | 15 | 0 | 1.3213 | 15-min pilot |
| `N11_lr2e-3.yaml` | 15 | 0, 1, 2 | 1.3140 (3 seeds) | 15-min pilot |
| `N11_lr4e-3.yaml` | 15 | 0 | 1.3204 | 15-min pilot |
| `N11_lr8e-3.yaml` | 15 | 0, 1, 2 | 1.3152 (3 seeds) | 15-min pilot |
| `N17_lr1e-3.yaml` | 15 | 0 | 1.3185 | 15-min pilot |
| `N17_lr2e-3.yaml` | 15 | 0, 1, 2 | 1.3074 (3 seeds) | 15-min pilot |
| `N17_lr4e-3.yaml` | 15 | 0, 1, 2 | 1.3057 (3 seeds) | 15-min pilot |
| `N17_lr5e-4.yaml` | 15 | 0 | 1.3574 | 15-min pilot |
| `N26_lr1e-3.yaml` | 15 | 0 | 1.3272 | 15-min pilot |
| `N26_lr1e-3_60min.yaml` | 60 | **not run** |  | Horizon-check option not needed: the runner-up at 15 min was 2e-3, so only {4e-3, 2e-3} were run (D13) |
| `N26_lr2e-3.yaml` | 15 | 0, 1, 2 | 1.3164 (3 seeds) | 15-min pilot |
| `N26_lr2e-3_60min.yaml` | 60 | 0 | 1.2077 | 60-min horizon check |
| `N26_lr4e-3.yaml` | 15 | 0, 1, 2 | 1.3141 (3 seeds) | 15-min pilot |
| `N26_lr4e-3_60min.yaml` | 60 | 0 | 1.2058 | 60-min horizon check; also counted as seed 0 of E02-N26 |
| `N26_lr5e-4.yaml` | 15 | 0 | 1.3650 | 15-min pilot |
| `N36_lr1e-3.yaml` | 15 | 0 | 1.3405 | 15-min pilot |
| `N36_lr2e-3.yaml` | 15 | 0, 1, 2 | 1.3282 (3 seeds) | 15-min pilot |
| `N36_lr4e-3.yaml` | 15 | 0, 1, 2 | 1.3241 (3 seeds) | 15-min pilot |
| `N36_lr5e-4.yaml` | 15 | 0 | 1.3791 | 15-min pilot |
| `N36_lr8e-3.yaml` | 15 | 0 | 2.0268 | Edge extension (N36's best was at the top of its grid) |
| `N49_lr1e-3.yaml` | 15 | 0 | 1.3774 | N49 pilots, added with N49 (rule P2b, D18-D19) |
| `N49_lr2e-3.yaml` | 15 | 0, 1, 2 | 1.3508 (3 seeds) | N49 pilots, added with N49 (rule P2b, D18-D19) |
| `N49_lr4e-3.yaml` | 15 | 0, 1, 2 | 1.3593 (3 seeds) | N49 pilots, added with N49 (rule P2b, D18-D19) |
| `N49_lr5e-4.yaml` | 15 | 0 | 1.4061 | N49 pilots, added with N49 (rule P2b, D18-D19) |
| `N65_lr1e-3.yaml` | 15 | 0 | 1.3919 | 15-min pilot |
| `N65_lr2e-3.yaml` | 15 | 0 | 1.3726 | 15-min pilot |
| `N65_lr4e-3.yaml` | 15 | 0 | 1.4203 | 15-min pilot |
| `N65_lr5e-4.yaml` | 15 | 0 | 1.4384 | 15-min pilot |
| `N7_lr1e-3.yaml` | 15 | 0 | 1.3414 | 15-min pilot |
| `N7_lr2e-3.yaml` | 15 | 0, 1, 2 | 1.3309 (3 seeds) | 15-min pilot |
| `N7_lr4e-3.yaml` | 15 | 0, 1, 2 | 1.3338 (3 seeds) | 15-min pilot |
| `N7_lr8e-3.yaml` | 15 | 0 | 1.3433 | 15-min pilot |
