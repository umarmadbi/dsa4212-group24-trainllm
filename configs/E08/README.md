# E08: the combined recipe (Phase 6)

The single changes that helped were combined into two recipes, each compared with the baseline E02-N49:

- **RA** = Muon + 6 x 768 + 16k vocabulary (`W6x768_V16384_muon_*`)
- **RB** = Muon + 11 x 576 + 16k vocabulary (`N49_V16384_muon_*`)

The tuned value is `muon_lr` (pilots at 1e-3, 2e-3, 4e-3, 8e-3; the `2.5e-4`, `5e-4`, `1.6e-2` and `3.2e-2` files are edge
extensions, not needed). The AdamW part (embeddings, LayerNorm) stays at 2e-3. The `emb*` pilots are a diagnostic (R1e):
RA at its picked Muon rate with the AdamW part at 1e-3 or 4e-3, to check whether that never-tuned rate leaves a gain; it
did not, and these never change the recipe. Only the picked rate of each recipe was run for 60 minutes, with 3 seeds.

**Outcome:** RA 1.1444 and RB 1.1531 on validation (-0.050 and -0.041 vs the baseline). RA vs RB (-0.009) is just inside
the declared threshold, so RB is the declared final recipe by the tie-break fixed in advance, and RA is reported as the best
measured. Both were then scored once on the test set (`configs/frozen_configs.yaml`).

- Write-up (question, design, rules, results, surprises): [docs/experiments/E08_final_recipe.md](../../docs/experiments/E08_final_recipe.md)
- Results: `results/runs/E08-*.json` ([results/runs/](../../results/runs/)); table with the declared checks: `python code/summarize.py E08`
- 48 configs, 12 run, 36 never run (options declared in advance that the rules did not call for; kept as the record of what was planned)

"BPB" is the final FineWeb-Edu validation bits per byte (mean over the seeds run). Minutes = counted training time.

| Config | Minutes | Seeds run | BPB | Role |
|---|---|---|---|---|
| `N49_V16384_muon_lr1.6e-2_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `N49_V16384_muon_lr1.6e-2_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `N49_V16384_muon_lr1e-3_15min.yaml` | 15 | 0 | 1.2923 | 15-min LR pilot |
| `N49_V16384_muon_lr1e-3_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `N49_V16384_muon_lr2.5e-4_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `N49_V16384_muon_lr2.5e-4_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `N49_V16384_muon_lr2e-3_15min.yaml` | 15 | 0, 1, 2 | 1.2735 (3 seeds) | 15-min LR pilot |
| `N49_V16384_muon_lr2e-3_60min.yaml` | 60 | 0, 1, 2 | 1.1531 (3 seeds) | **The pick**: 60-min final |
| `N49_V16384_muon_lr3.2e-2_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `N49_V16384_muon_lr3.2e-2_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `N49_V16384_muon_lr4e-3_15min.yaml` | 15 | 0, 1, 2 | 1.2841 (3 seeds) | 15-min LR pilot |
| `N49_V16384_muon_lr4e-3_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `N49_V16384_muon_lr5e-4_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `N49_V16384_muon_lr5e-4_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `N49_V16384_muon_lr8e-3_15min.yaml` | 15 | 0 | 1.3073 | 15-min LR pilot |
| `N49_V16384_muon_lr8e-3_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `W6x768_V16384_muon_lr1.6e-2_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `W6x768_V16384_muon_lr1.6e-2_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `W6x768_V16384_muon_lr1.6e-2_emb1e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr1.6e-2_emb4e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr1e-3_15min.yaml` | 15 | 0, 1, 2 | 1.2638 (3 seeds) | 15-min LR pilot |
| `W6x768_V16384_muon_lr1e-3_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `W6x768_V16384_muon_lr1e-3_emb1e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr1e-3_emb4e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr2.5e-4_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `W6x768_V16384_muon_lr2.5e-4_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `W6x768_V16384_muon_lr2.5e-4_emb1e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr2.5e-4_emb4e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr2e-3_15min.yaml` | 15 | 0 | 1.2645 | 15-min LR pilot |
| `W6x768_V16384_muon_lr2e-3_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `W6x768_V16384_muon_lr2e-3_emb1e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr2e-3_emb4e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr3.2e-2_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `W6x768_V16384_muon_lr3.2e-2_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `W6x768_V16384_muon_lr3.2e-2_emb1e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr3.2e-2_emb4e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr4e-3_15min.yaml` | 15 | 0, 1, 2 | 1.2627 (3 seeds) | 15-min LR pilot |
| `W6x768_V16384_muon_lr4e-3_60min.yaml` | 60 | 0, 1, 2 | 1.1444 (3 seeds) | **The pick**: 60-min final |
| `W6x768_V16384_muon_lr4e-3_emb1e-3_15min.yaml` | 15 | 0 | 1.2702 | R1e diagnostic |
| `W6x768_V16384_muon_lr4e-3_emb4e-3_15min.yaml` | 15 | 0 | 1.2659 | R1e diagnostic |
| `W6x768_V16384_muon_lr5e-4_15min.yaml` | 15 | **not run** |  | Edge extension, not needed |
| `W6x768_V16384_muon_lr5e-4_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `W6x768_V16384_muon_lr5e-4_emb1e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr5e-4_emb4e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr8e-3_15min.yaml` | 15 | 0 | 1.3896 | 15-min LR pilot |
| `W6x768_V16384_muon_lr8e-3_60min.yaml` | 60 | **not run** |  | Unused: not the picked rate |
| `W6x768_V16384_muon_lr8e-3_emb1e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
| `W6x768_V16384_muon_lr8e-3_emb4e-3_15min.yaml` | 15 | **not run** |  | R1e diagnostic: only run at RA's pick (4e-3) |
