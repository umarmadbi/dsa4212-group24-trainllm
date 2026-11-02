# code/plots/

Scripts that turn files in `results/` into the figures in `report/figures/`. **Empty for now.** Figures are added as
experiments finish.

Rules for figure scripts (so every figure in the report can be regenerated and traced):

- Read only from `results/runs/*.json` and `results/compute_ledger.csv`; never type numbers in by hand.
- Write to `report/figures/<name>.png` (or `.pdf`).
- One script per figure, named after it, e.g. `fig_size_vs_bpb.py` -> `report/figures/size_vs_bpb.pdf`.
- Add a row to the table below.

| Script | Reads | Writes | Shows |
|---|---|---|---|
| | | | |
