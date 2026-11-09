# E05: Muon vs AdamW (Phase 4a)

N49 trained with the Muon optimiser on its hidden weight matrices (AdamW for embeddings and LayerNorm) instead of AdamW
everywhere. Muon got the same tuning effort AdamW got in E01: 4 learning rates in 15-minute pilots, close calls with 3
seeds, then a 60-minute horizon check at the neighbouring rates before the 3-seed final. The two AdamW 60-minute configs
were a declared fairness check, to be run only if Muon's horizon check changed its pick (it did not).

**Outcome:** Muon beats a tuned AdamW by 0.026 BPB in the same hour (1.1683 vs 1.1945), on every validation set. The
extra matrix multiplications cost only about 0.002 of that gain.

- Write-up (question, design, rules, results, surprises): [docs/experiments/E05_muon.md](../../docs/experiments/E05_muon.md)
- Results: `results/runs/E05-*.json` ([results/runs/](../../results/runs/)); table with the declared checks: `python code/summarize.py E05`
- 10 configs, 7 run, 3 never run (options declared in advance that the rules did not call for; kept as the record of what was planned)

"BPB" is the final FineWeb-Edu validation bits per byte (mean over the seeds run). Minutes = counted training time.

| Config | Minutes | Seeds run | BPB | Role |
|---|---|---|---|---|
| `N49_adamw_lr1e-3_60min.yaml` | 60 | **not run** |  | Conditional AdamW check (D30): not needed, Muon's pick held |
| `N49_adamw_lr4e-3_60min.yaml` | 60 | **not run** |  | Conditional AdamW check (D30): not needed, Muon's pick held |
| `N49_muon_lr1e-3_15min.yaml` | 15 | 0 | 1.3086 | 15-min LR pilot |
| `N49_muon_lr1e-3_60min.yaml` | 60 | 0 | 1.1716 | Horizon check (neighbour of the pick) |
| `N49_muon_lr2e-3_15min.yaml` | 15 | 0, 1, 2 | 1.2914 (3 seeds) | 15-min LR pilot |
| `N49_muon_lr2e-3_60min.yaml` | 60 | 0, 1, 2 | 1.1683 (3 seeds) | **The pick**: 3-seed final, compared with E02-N49 |
| `N49_muon_lr4e-3_15min.yaml` | 15 | 0, 1, 2 | 1.2934 (3 seeds) | 15-min LR pilot |
| `N49_muon_lr4e-3_60min.yaml` | 60 | 0 | 1.1689 | Horizon check (neighbour of the pick) |
| `N49_muon_lr8e-3_15min.yaml` | 15 | 0 | 1.3167 | 15-min LR pilot |
| `N49_muon_lr8e-3_60min.yaml` | 60 | **not run** |  | Not a neighbour of the pick (2e-3), so not part of the horizon check |
