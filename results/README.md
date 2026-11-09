# results/

Everything the GPU and CPU jobs produced. `modal_app.py` writes these files automatically when a job finishes; nothing
here is typed by hand. **Never edit or delete them**: every number in the report must be traceable to a file here.
Read them with `python code/summarize.py <experiment>` (a table per experiment, e.g. `E05`, or `TEST`), or open the JSON
files directly. The write-ups that interpret them are in [docs/experiments/](../docs/experiments/).

| Path | Written by | Contents |
|---|---|---|
| [`runs/`](runs/) | `modal_app.py launch` / `recover` | **One JSON file per training run** (180), plus `_replaced/` (older copies of 5 runs) |
| [`rescore/`](rescore/) | `modal_app.py rescore` | 27 validation re-scores of saved models (contamination-free, extra sets, overlapping windows) |
| [`test/`](test/) | `modal_app.py test_eval` | **The one-time test scores** of the 57 frozen models |
| [`failed/`](failed/) | `modal_app.py launch` | Error trace of the one run that crashed on the GPU |
| `compute_ledger.csv` | `modal_app.py` (every job) | **One row per GPU or CPU job of any kind**, including failed and repeated ones. The source of truth for compute time and cost |
| `contamination.json` | `modal_app.py contamination` | 13-gram overlap of every validation and test document with each training corpus (D25) |

## compute_ledger.csv

270 rows, one per job: training runs (`kind` = `pilot` 145, `final` 91, `benchmark` 24, `smoke` 3), data preparation on
CPU (3) and test scoring (4). Columns: `run_id`, `experiment`, `kind`, `date_sgt`, `member`, `modal_workspace`,
`gpu_name`, `git_commit`, `config`, `seed`, `counted_train_s` (the budgeted training time), `wallclock_s` (container
time), `tokens_processed`, `status`, `cost_usd` (estimated from container time at $0.80 per L4 hour), `notes`. The
estimated total is about $79.

| `status` | Meaning |
|---|---|
| `completed` | Finished normally (194 rows) |
| `completed (result transfer failed)` | Trained normally, but the result did not reach the laptop; recovered later with `recover` (8 rows, E00) |
| `failed` | The job, or the laptop's connection to it, failed; the reason is in `notes` (68 rows) |

Most `failed` rows are not failed training. 59 record the laptop losing its connection to Modal while it waited for a
job; the job kept running on Modal and its result was recovered later (and has its own `completed` row). 8 record a
YAML parsing bug in the first benchmark launch (`1.0e9` read as text; nothing trained). 1 is the resume test's
GPU-only crash, which was fixed and rerun (see `failed/`). A few rows have an estimated cost written as text
(e.g. `~0.10 (est.)`), and two use the workspace name `umarmadbi/main` for `umarmadbi`; scripts reading the ledger
should allow for both.
