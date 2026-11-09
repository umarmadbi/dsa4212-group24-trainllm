# E00: pipeline checks and speed benchmarks (Phase 0)

Preparation only; nothing here is reported as a result. Short runs that measured how fast each model size trains on
the L4 (which set the size grid and showed how much data to download), whether `torch.compile` and a larger micro-batch
pay off, and whether a preempted run resumes correctly on Modal. The scores of these runs are meaningless (5 minutes of
training); only their speed and memory matter.

E00 used older size names: S1 = N7, S2 = N17, S3 = N36, S4 = N65; S5 (91.6M) and S6 (123.6M) were too slow and were
dropped from the size grid. These runs started from uncommitted code (`git_dirty: true` in their result files), which is
why `launch` now refuses to start from uncommitted code (D07).

- Write-up (question, design, rules, results, surprises): [docs/experiments/E00_pipeline_and_benchmarks.md](../../docs/experiments/E00_pipeline_and_benchmarks.md)
- Results: `results/runs/E00-*.json` ([results/runs/](../../results/runs/)); table with the declared checks: `python code/summarize.py E00`
- 9 configs, 9 run

"BPB" is the final FineWeb-Edu validation bits per byte (mean over the seeds run). Minutes = counted training time.

| Config | Minutes | Seeds run | BPB | Role |
|---|---|---|---|---|
| `bench_S1.yaml` | 5 | 0 | 1.4947 | 5-minute speed benchmark |
| `bench_S1_micro64.yaml` | 5 | 0 | 1.4992 | S1 with micro-batch 64: not faster, so micro-batch 32 is kept |
| `bench_S2.yaml` | 5 | 0 | 1.5466 | 5-minute speed benchmark |
| `bench_S3.yaml` | 5 | 0 | 1.7050 | 5-minute speed benchmark |
| `bench_S3_nocompile.yaml` | 5 | 0 | 1.7370 | S3 without `torch.compile`: compile is worth its warm-up (+20 % speed) |
| `bench_S4.yaml` | 5 | 0 | 1.8452 | 5-minute speed benchmark |
| `bench_S5.yaml` | 5 | 0 | 1.9047 | 5-minute speed benchmark |
| `bench_S6.yaml` | 5 | 0 | 2.0087 | 5-minute speed benchmark |
| `resume_test_N7.yaml` | 4 | 0 | 1.4675 | Kills itself once at step 1500; Modal restarted it and it resumed with its counted time kept (the first attempt exposed a GPU-only resume bug, see `results/failed/`) |
