# E04: repeated data (Phase 3b)

All other runs read fresh text for the whole hour. Here N49 gets its usual hour but only a small random slice of
FineWeb-Edu, sized so that the hour reads it about 2, 4, 8, 16 or 32 times. Two follow-ups were declared after seeing the
results: other model sizes on the same 36M-token slice (F2: is a smaller model better when unique data is limited?) and
the same slice with window boundaries shifted every pass (F3: does seeing identical samples cause the damage?).

**Outcome:** 2 passes are indistinguishable from fresh data; 4 passes are measurably worse (+0.024); 8 or more passes are
much worse, and at 32 passes the model memorises its 9M tokens (validation 3.64 BPB, worse than an untrained model).
F2: the biggest model (N65) was best on the limited slice, the opposite of the prediction. F3: shifting the windows
removes about a quarter of the damage.

- Write-up (question, design, rules, results, surprises): [docs/experiments/E04_repetition.md](../../docs/experiments/E04_repetition.md)
- Results: `results/runs/E04-*.json` ([results/runs/](../../results/runs/)); table with the declared checks: `python code/summarize.py E04`
- 9 configs, 9 run

"BPB" is the final FineWeb-Edu validation bits per byte (mean over the seeds run). Minutes = counted training time.

| Config | Minutes | Seeds run | BPB | Role |
|---|---|---|---|---|
| `N17_U36M.yaml` | 60 | 0 | 1.3036 | Follow-up F2 (D26): N17 on the same 36.25M-token slice as rep8 |
| `N26_U36M.yaml` | 60 | 0 | 1.2913 | Follow-up F2 (D26): N26 on the same slice |
| `N49_rep16.yaml` | 60 | 0 | 1.6648 | N49, data repeated ~16 times |
| `N49_rep2.yaml` | 60 | 0, 1, 2 | 1.2011 (3 seeds) | N49, data repeated ~2 times |
| `N49_rep32.yaml` | 60 | 0 | 3.6424 | N49, data repeated ~32 times |
| `N49_rep4.yaml` | 60 | 0, 1, 2 | 1.2189 (3 seeds) | N49, data repeated ~4 times |
| `N49_rep8.yaml` | 60 | 0, 1, 2 | 1.2842 (3 seeds) | N49, data repeated ~8 times |
| `N49_rep8_shift.yaml` | 60 | 0 | 1.2709 | Follow-up F3 (D26): boundaries move every pass |
| `N65_U36M.yaml` | 60 | 0 | 1.2688 | Follow-up F2 (D26): N65 on the same slice |
