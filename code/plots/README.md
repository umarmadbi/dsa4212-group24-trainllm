# code/plots/

Scripts that turn files in `results/` into the figures in `report/figures/`. Run any script from the repository root or this folder, e.g. `python code/plots/fig_size_vs_bpb.py` (needs numpy and
matplotlib only).

Rules for figure scripts (so every figure in the report can be regenerated and traced):

- Read only from `results/runs/*.json` and `results/compute_ledger.csv`; never type numbers in by hand.
- Write to `report/figures/<name>.png` (or `.pdf`).
- One script per figure, named after it, e.g. `fig_size_vs_bpb.py` -> `report/figures/size_vs_bpb.pdf`.
- Add a row to the table below.

| Script | Reads | Writes | Shows |
|---|---|---|---|
| `_common.py` | - | - | Shared loaders (run files, test scores), palette and style |
| `fig_size_vs_bpb.py` | `runs/E01-*`, `runs/E02-*` | `size_vs_bpb.png/.pdf` | Val BPB vs model size at 15 and 60 min, tokens per parameter (E01, E02) |
| `fig_repetition.py` | `runs/E02-N49*`, `runs/E04-*` | `repetition.png/.pdf` | Final BPB vs passes over a repeated subset; validation curves against passes (E04) |
| `fig_muon_curves.py` | `runs/E02-N49*`, `runs/E05-*_60min*` | `muon_curves.png/.pdf` | Muon vs AdamW learning curves and their gap at equal time / equal tokens (E05) |
| `fig_recipe_attribution.py` | `runs/E02,E05-E08`, `test/` | `recipe_attribution.png/.pdf` | Each change alone and combined, val and test, vs the additive prediction (E08) |
| `fig_val_vs_test.py` | `runs/`, `test/` | `val_vs_test.png/.pdf` | Validation vs test BPB for the 57 frozen models (D47) |
| `fig_recipe_and_test.py` | `runs/`, `test/` | `recipe_and_test.png/.pdf` | Figures 4 and 5 combined into two panels (used in the page-limited report) |
