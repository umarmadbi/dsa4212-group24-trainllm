# report/

The final report, written in LaTeX with the professor's Overleaf template. **Not started yet**; writing is planned
for 11-13 November.

| Path | What it will contain |
|---|---|
| `report.tex` | The report source (8 pages; no code in the main text) |
| `references.bib` | Bibliography |
| `figures/` | Figures, each made by a script in `code/plots/` from files in `results/` |

Rules for the report:

- Names, student numbers and the GitHub URL at the top.
- The compute budget, initialisation and evaluation objective are declared at the start (copied from
  [`docs/DECLARATIONS.md`](../docs/DECLARATIONS.md)).
- Every number must come from a file in `results/` (an AI agent should be able to match each one to a run).
- Validation results for decisions; test results only for the final frozen configurations.
- Differences smaller than the measured seed noise are reported as inconclusive, not as wins.
