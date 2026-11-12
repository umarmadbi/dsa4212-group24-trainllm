# report/

The final report: complete LaTeX source and the figures it uses. The submitted PDF (10 pages, on Canvas) is compiled
from `report.tex`.

| Path | What it contains |
|---|---|
| `report.tex` | Complete report source (title block, sections 1-8, references, use of AI tools). The references are written out at the end of the file, so no separate `.bib` file is needed |
| `figures/` | Every figure, as `.pdf` (used by `report.tex`) and `.png`. Each is made by one script in [`code/plots/`](../code/plots/) from files in [`results/`](../results/) |

## Compiling

```
cd report
pdflatex report.tex
pdflatex report.tex      # second pass resolves section, table and figure references
```

Or upload `report.tex` and the four figure PDFs it uses (keeping them in a `figures/` folder) to Overleaf. Only
standard packages are used (amsmath, graphicx, booktabs, tabularx, caption, hyperref, xurl).

## Figures

| Figure in report | File | Made by |
|---|---|---|
| Figure 1 (model size) | `figures/size_vs_bpb.pdf` | `code/plots/fig_size_vs_bpb.py` |
| Figure 2 (repetition) | `figures/repetition.pdf` | `code/plots/fig_repetition.py` |
| Figure 3 (Muon vs AdamW) | `figures/muon_curves.pdf` | `code/plots/fig_muon_curves.py` |
| Figure 4 (recipes and test set) | `figures/recipe_and_test.pdf` | `code/plots/fig_recipe_and_test.py` |
| Not in the report | `figures/recipe_attribution.*`, `figures/val_vs_test.*` | The two panels of Figure 4 as separate figures (`fig_recipe_attribution.py`, `fig_val_vs_test.py`) |

## Where each number comes from

Every number in the report can be regenerated from `results/` (run from the repository root):

| Report section | Command | Notes |
|---|---|---|
| 2 Declared setup (seed noise, machine speed) | `python code/summarize.py E02` | Declarations themselves: [`docs/DECLARATIONS.md`](../docs/DECLARATIONS.md) |
| 4.2 Protocol (thresholds, job counts, cost) | `python code/summarize.py E02`; `results/compute_ledger.csv` | |
| 5.1 Model size, Equation 1 | `python code/summarize.py E02`; `python code/fit_scaling_law.py` | The fit needs scipy (pinned in `setup/requirements-dev.txt`) |
| 5.2 Data quality, Table 2 | `python code/summarize.py E03` | |
| 5.3 Repetition | `python code/summarize.py E04`; `python code/fit_repetition.py` | |
| 5.4 Muon | `python code/summarize.py E05` | |
| 5.5 Shape and vocabulary, Table 3 | `python code/summarize.py E06`; `python code/summarize.py E07` | |
| 5.6 Recipes and test set | `python code/summarize.py E08`; `python code/summarize.py TEST` | |

Table 1 (predictions) summarises the decision log, [`docs/decisions.md`](../docs/decisions.md), where each prediction was
written before its experiment ran.

## Rules the report follows

- Names, student numbers and the GitHub URL at the top.
- The compute budget, initialisation and evaluation objective are declared at the start (from
  [`docs/DECLARATIONS.md`](../docs/DECLARATIONS.md)).
- Every number comes from a file in `results/`.
- Validation results for decisions; test results only for the final frozen configurations.
- Differences smaller than the measured seed noise are reported as inconclusive, not as wins.
