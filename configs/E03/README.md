# E03: data quality (Phase 3a)

Same model (N49), same hour, same tokenizer; only the training text changes: FineWeb-Edu (the baseline, run as
`configs/E02/N49.yaml`), raw FineWeb, and DCLM. A model always looks best on its own kind of text, so the declared judge is
the average BPB over four validation sets (FineWeb-Edu, FineWeb, DCLM, Wikipedia). Short pilots checked that N49's
learning rate, tuned on FineWeb-Edu, is not clearly wrong for the other two corpora. Because DCLM won, a follow-up (R8)
trained on a 60/40 FineWeb-Edu/DCLM mixture.

**Outcome:** DCLM beats FineWeb-Edu and FineWeb on the four-set average by about 0.028 (real); FineWeb-Edu vs FineWeb is
a tie. The mixture beats all three on the average, at a cost of +0.022 on FineWeb-Edu itself. The ranking holds without
contaminated validation documents and after adjusting for machine speed.

- Write-up (question, design, rules, results, surprises): [docs/experiments/E03_data_quality.md](../../docs/experiments/E03_data_quality.md)
- Results: `results/runs/E03-*.json` ([results/runs/](../../results/runs/)); table with the declared checks: `python code/summarize.py E03`
- 7 configs, 7 run

"BPB" is the final FineWeb-Edu validation bits per byte (mean over the seeds run). Minutes = counted training time.

| Config | Minutes | Seeds run | BPB | Role |
|---|---|---|---|---|
| `N49_dclm.yaml` | 60 | 0, 1, 2 | 1.2763 (3 seeds) | 60-min arm |
| `N49_dclm_lr2e-3_15min.yaml` | 15 | 0 | 1.4453 | LR check (rule R6, D25) |
| `N49_dclm_lr4e-3_15min.yaml` | 15 | 0 | 1.4444 | LR check (rule R6, D25) |
| `N49_fineweb.yaml` | 60 | 0, 1, 2 | 1.2680 (3 seeds) | 60-min arm |
| `N49_fineweb_lr2e-3_15min.yaml` | 15 | 0 | 1.4278 | LR check (rule R6, D25) |
| `N49_fineweb_lr4e-3_15min.yaml` | 15 | 0 | 1.4262 | LR check (rule R6, D25) |
| `N49_mix_fwe60_dclm40.yaml` | 60 | 0, 1, 2 | 1.2167 (3 seeds) | Follow-up R8 (D26): 60 % FineWeb-Edu + 40 % DCLM by tokens |
