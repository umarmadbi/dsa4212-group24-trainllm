# results/runs/

**One JSON file per training run**, written when the run finishes (or later by `modal_app.py recover`, if the laptop lost
its connection). The file name is the config's `run_id` plus the seed: `configs/E02/N49.yaml` with seed 0 ->
`E02-N49__s0.json`. 180 files, one for every (config, seed) pair that was run. Which configs were run, and with which
seeds, is listed in each experiment's README under [`configs/`](../../configs/).

Summarise them with `python code/summarize.py E02` (one experiment) from the repository root.

| Folder | Contents |
|---|---|
| [`_replaced/`](_replaced/) | Older copies of 5 result files that were later written again. Kept, never deleted |

## What is in a run file

| Key | Meaning |
|---|---|
| `run_id`, `experiment`, `seed`, `config` | Which run this is and every setting it used |
| `meta` | Git commit, whether the code was uncommitted (`git_dirty`), SHA-256 of the code and config, who launched it, Modal account, GPU name |
| `params` | Parameter counts: total, embeddings, non-embedding |
| `final_val_bpb` | **The headline number**: bits per byte on the full FineWeb-Edu validation split, final model (`null` if the run diverged) |
| `final_val` | Full scoring details per validation set (loss per token, bytes, tokens scored) |
| `curve` | Learning curve: validation BPB on a 1 MB subset every 5 % of the budget |
| `counted_train_s`, `budget_s` | Counted training time and the budget it was given |
| `tokens_processed`, `steps`, `passes_over_data` | How much text the model read |
| `avg_tokens_per_s`, `steady_tokens_per_s`, `steady_tokens_per_s_early` / `_late` | Speed; early vs late shows any slow-down within the run |
| `first_step_s` | Time of the first step = the compile warm-up (should be 17-45 s; ~0.1 s means the run skipped it, which is a fault) |
| `attempts` | One entry per (re)start; more than one means the run was interrupted and resumed |
| `diverged`, `diverged_at_step` | Whether the loss blew up (NaN) |
| `frac_steps_clipped`, `train_log` | Gradient clipping and the training log (loss, learning rate, gradient norm every 30 s) |
| `gpu_telemetry`, `gpu_telemetry_summary` | GPU clock, temperature and power every 30 s (L4s run at their 72 W power cap) |
| `peak_memory_gb`, `mfu_steady`, `versions` | Memory, GPU utilisation, software versions |
