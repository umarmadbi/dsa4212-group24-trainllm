# E02: which model size is best in 60 minutes (Phase 2)

The main experiment. Seven sizes from 7M to 65M parameters, each trained for the full 60 minutes at its E01 learning
rate; everything else identical. Close calls were repeated with seeds 1 and 2. N26 was run with four more seeds to measure
seed noise (its seed 0 is the E01 horizon run `E01-N26_lr4e-3_60min`). N49 (11 layers x 576) was added between N36 and
N65 because the fitted optimum fell in that gap (rule P2b); it became the baseline for every later experiment.

**Outcome:** N36, N49 and N65 are tied within noise (1.19-1.20 BPB) and clearly better than the smaller sizes. They read
only 4-11 training tokens per parameter in the hour, far below Chinchilla's ~20. Seed-to-seed noise at 60 minutes is
about 0.006 BPB for single runs, mostly from some cloud GPUs being faster than others.

Every E02 config scores the final model on all four validation sets (`eval_corpora`); FineWeb-Edu is the headline.

- Write-up (question, design, rules, results, surprises): [docs/experiments/E02_model_size.md](../../docs/experiments/E02_model_size.md)
- Results: `results/runs/E02-*.json` ([results/runs/](../../results/runs/)); table with the declared checks: `python code/summarize.py E02`
- 9 configs, 7 run, 2 never run (options declared in advance that the rules did not call for; kept as the record of what was planned)

"BPB" is the final FineWeb-Edu validation bits per byte (mean over the seeds run). Minutes = counted training time.

| Config | Minutes | Seeds run | BPB | Role |
|---|---|---|---|---|
| `N11.yaml` | 60 | 0 | 1.2426 | 60-min final |
| `N17.yaml` | 60 | 0 | 1.2296 | 60-min final |
| `N26.yaml` | 60 | 1, 2, 3, 4 | 1.2139 (4 seeds) | Seeds 1-4 = noise calibration; seed 0 is the E01 horizon run |
| `N36.yaml` | 60 | 0, 1, 2 | 1.1959 (3 seeds) | 60-min final |
| `N4.yaml` | 60 | **not run** |  | Edge extension, run only if the smallest size had won (it did not) |
| `N49.yaml` | 60 | 0, 1, 2 | 1.1945 (3 seeds) | Added by rule P2b (declared D18, triggered D19); **the baseline** for E03-E08 (D22) |
| `N65.yaml` | 60 | 0, 1, 2 | 1.1951 (3 seeds) | 60-min final |
| `N7.yaml` | 60 | 0 | 1.2824 | 60-min final |
| `N92.yaml` | 60 | **not run** |  | Edge extension, run only if the largest size had won (it did not) |
