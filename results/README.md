# results/

Everything the GPU runs produced. `launch` and `recover` in `modal_app.py` write these files automatically when a
run finishes, so nothing here is typed by hand. **Never edit or delete them**: every number in the report must be
traceable to a file here.

| Path | Written by | Contents |
|---|---|---|
| `compute_ledger.csv` | `modal_app.py` (every launch) | **One row per GPU job of any kind**, including failed and replaced ones: run ID, date, who launched it, Modal account, GPU, git commit, counted training time, container time, tokens, status, estimated cost, notes. The source of truth for how much compute we used |
| `runs/<run_id>__s<seed>.json` | `code/train.py` (returned via `launch`) | **The full result of one run**: see "What is in a run file" below |
| `runs/_replaced/` | `modal_app.py launch --fresh` | Older results of runs that were rerun (e.g. because of an infrastructure fault), kept with a timestamp, never deleted |
| `failed/<run_id>.txt` | `modal_app.py` | Python error trace of a run that crashed |
| `rescore/<run_id>__<tag>.json` | `modal_app.py rescore` | Validation re-scores of saved models: without contaminated documents (`nocontam`), extra sets (`full`), overlapping windows (`slide128`) |
| `test/<run_id>__s<seed>.json` | `modal_app.py test_eval` (Phase 6, D42) | **The one-time test score** of a frozen model: per test set the declared BPB, the same without contaminated documents, overlapping windows (core models), text samples (marked runs), and the hash/commit of the frozen list that allowed it. Written once; never recomputed |
| `contamination.json` | `modal_app.py contamination` | 13-gram overlap of every validation and test document with each training corpus (D25) |

Read them with `python code/summarize.py E01` (a table per experiment), or open a JSON file directly.

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

## Ledger `status` values

| Status | Meaning |
|---|---|
| `completed` | Finished normally |
| `diverged` | Finished early because the loss became NaN (a valid result: that learning rate is too high) |
| `failed` | Crashed or never finished; reason in `notes` and `failed/` |
| `completed (result transfer failed)` | Trained fine but the result did not reach the laptop; recovered later with `recover` (E00 only) |

Cost is estimated from container time at $0.80 per L4 hour.
