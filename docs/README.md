# docs/

The written record of the project: the rules fixed before the experiments, what each experiment asked and found, why
every choice was made, and when each thing happened. The aim is that anyone can trace every number in the report back to
a run, its config and its result file, including failed, aborted and replaced runs.

**Start with [OVERVIEW.md](OVERVIEW.md).**

| File | What it is |
|---|---|
| [`OVERVIEW.md`](OVERVIEW.md) | Plain-language account of the whole project: the question, how a run works, what each phase found, what did not work, what remains uncertain, glossary |
| [`DECLARATIONS.md`](DECLARATIONS.md) | **The rules fixed before any experiment** (compute budget, initialisation, data, tokenizer, evaluation, seeds, test-set lock), committed on 1 Nov 2026, followed by an Amendments table: every later change, dated, with its reason |
| [`REPRODUCE.md`](REPRODUCE.md) | How to rebuild the data, rerun any experiment and the evaluation checks, and which configs and commands produce each result |
| [`experiments/`](experiments/) | One write-up per experiment (E00-E08): question, prediction, design, decision rules, gaps checked, runs, results, findings, surprises |
| [`decisions.md`](decisions.md) | The decision log, D01-D47 in order: for each decision the context, the options and gaps considered, the choice and why. Experiment docs and commit messages refer to these numbers |
| [`experiment_log.md`](experiment_log.md) | Chronological lab log: what was launched or found, hour by hour, 30 Oct - 6 Nov 2026 |

## How the records were kept

- Each experiment's question, prediction, design and decision rules were written **before** its runs, and committed with
  its configs; results and findings were added after. A prediction that missed is kept and marked as missed.
- Every rule change after the first experiment is an amendment in `DECLARATIONS.md`, recorded before the runs it affects.
- Times are Singapore time (UTC+8). `Dnn` refers to an entry in `decisions.md`; `E0n` to an experiment; `R1`, `P2b`, `F2`
  and similar are rule or follow-up labels defined in that experiment's write-up.
- Commit hashes in the records (e.g. `afb895c`) identify the exact code each batch of runs used; each result file stores
  the same commit.
- `decisions.md` and `experiment_log.md` are working records written during the project, so they are terse and include
  intermediate states that later entries correct. The experiment write-ups and `OVERVIEW.md` give the final picture.
