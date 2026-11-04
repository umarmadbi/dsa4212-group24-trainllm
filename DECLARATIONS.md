# Declarations

DSA4212 group project: *Training small language models under computational constraints*

This file fixes the budget, initialisation, data, evaluation and run policies **before any final training run**. Its git commit timestamp is the evidence that these choices were declared in advance, as the project brief requires. Any later change is recorded under [Amendments](#amendments) with a date and a reason; no amendment is allowed after the first final run except to fix a bug, which is then disclosed in the report.

Declared: 30 October 2026.

---

## 1. Compute budget

| Item | Declaration |
|---|---|
| Hardware | 1 × NVIDIA L4 (24 GB) on Modal (`gpu="L4"`), standard (preemptible) pricing |
| Budget per final run | **60 minutes (3,600 s) of counted training time** |
| Software | Python, PyTorch with CUDA; exact versions pinned in `requirements.txt` and logged in every run |
| Precision | bf16 mixed precision |

### What counts toward the 60 minutes

| Counts | Excluded (logged separately) |
|---|---|
| Forward pass, backward pass, optimizer step | Modal container start-up and image build |
| Data loading inside the training loop | Data download, splitting and tokenization (done once, beforehand) |
| `torch.compile` warm-up, if compile is used | Model construction and optimizer creation |
| Steps re-run after a preemption (the lost first attempt is not counted, but logged) | Periodic validation and checkpoint saving (clock paused) |
| | Final validation and test evaluation after training |

### How the clock is measured

- A monotonic timer starts immediately before the first training step.
- `torch.cuda.synchronize()` is called before every clock reading, so asynchronous GPU work is fully counted.
- The clock is paused during periodic validation and checkpoint saving.
- Training stops at the first step boundary at or after 3,600 s of counted time. The actual counted time (including the overshoot of under one step) is recorded.

### Preemption and resume

- A checkpoint is saved about every 10 minutes of counted time. It includes model, optimizer, step count, cumulative counted time, cumulative tokens, RNG states and data-loader position.
- A resumed run continues from the saved cumulative time; resuming never resets the budget.
- Steps lost between the last checkpoint and an interruption are re-run. The lost wall-clock time is recorded in the compute ledger.

### Learning-rate schedule (a declared modification of nanoGPT)

- Linear warmup over a fixed number of steps, then cosine decay to a minimum learning rate.
- Decay progress is the **fraction of the 3,600 s budget consumed**, not a planned step count. Every run therefore finishes its decay exactly at the cap, regardless of its throughput.

### Preliminary work (excluded from the per-run budget)

- Code debugging and 1–2 minute smoke tests (Colab or local; never reported as results).
- Throughput benchmarks and learning-rate pilots on the Modal L4.
- Tokenizer training on Modal CPU.

### Learning-rate tuning protocol (same effort for every configuration)

- Each model size, and each optimiser in the optimiser experiment, gets **the same tuning effort**: 4 peak learning rates on a log grid (factor 2 apart), each run for 15 minutes with the time-based schedule compressed to 15 minutes. The best value on the full validation split is used for that configuration's final run. If the best value is at the edge of the grid, one more value beyond the edge is tried.
- **Horizon check.** The best learning rate can shift with run length, so for the baseline model size the top two pilot values are both run for the full 60 minutes. If they disagree with the 15-minute ranking, all sizes are shifted one grid step in the same direction, and this is reported.
- Batch size, context length (512), and whether `torch.compile` is used are fixed from Phase 0–1 pilots and recorded in the Amendments table before the first final run.

All preliminary GPU time is recorded in the compute ledger (`results/compute_ledger.csv`).

### Running across several Modal accounts

GPU credits come from up to four group members' Modal accounts (Starter plan, $30/month each). Runs are distributed across these accounts. This does not change the declared hardware: **every final run uses one L4 GPU and the same code, data and budget, whichever account it is billed to.**

- **One run, one GPU.** A single run never uses more than one GPU at a time. A preempted run may resume on another L4 (same or different account); its counted time carries over.
- **Same code.** Every run is launched from a committed git version. The commit hash is logged, and runs whose commit or library versions differ from the rest of their comparison are flagged.
- **Same data.** Each account holds its own copy of the prepared data in a Modal Volume, produced by the same deterministic preparation script. Before any final run, the SHA-256 hashes of every split and token file in each account's Volume must match `data/manifest.json`.
- **Same hardware.** The GPU name reported by `nvidia-smi` is logged for every run and must be an L4. The throughput rule (§6) catches a slow machine on any account.
- **How jobs are launched.** Each member either launches runs from their own account, or invites the lead member into their Modal workspace so jobs can be launched there with a separate login profile and billed to that workspace's credits.
- **Accounting.** The compute ledger records which account (workspace) ran and paid for each run.

Planned cost: about $50 for ~43 one-hour final runs plus pilots and evaluation, out of about $120 of pooled credits. The remainder is a reserve for reruns and follow-up experiments.

---

## 2. Initialisation

- Every model starts from **random initialisation**, following nanoGPT's scheme (normal init with depth-scaled residual projections), seeded as in §6.
- **No pretrained weights, embeddings or other pretrained neural components** are used at any point.
- A fresh model and optimizer are created for every independent final run. Each run writes to its own output directory, so no stale checkpoint can be loaded by mistake.
- The tokenizer (§4) is a byte-pair-encoding merge table learned from our training text by a counting algorithm. It contains no neural network.

**Data disclosure.** Our corpora (§3) were *selected* by their publishers using quality classifiers (FineWeb-Edu: a classifier trained on Llama3-70B-Instruct annotations; DCLM: a fastText classifier). The text itself is human-written web text, and those classifiers are never part of, nor used to initialise, our models. We treat these as documented prepared corpora.

---

## 3. Data

### Corpora

| Role | Corpus | Source | Licence |
|---|---|---|---|
| Main training corpus | FineWeb-Edu, `sample-10BT` | `HuggingFaceFW/fineweb-edu` | ODC-By |
| Data-quality comparison arm | FineWeb (unfiltered), `sample-10BT` | `HuggingFaceFW/fineweb` | ODC-By |
| Data-quality comparison arm | DCLM-baseline 1.0 | `mlfoundations/dclm-baseline-1.0` | CC-BY-4.0 |
| Out-of-domain evaluation only | English Wikipedia (held-out articles) | `wikimedia/wikipedia` | CC-BY-SA |

Only a subset of each corpus is downloaded. **Each training corpus (FineWeb-Edu, FineWeb, DCLM) gets at least 1.5× the number of tokens the fastest configuration processes in 60 minutes** (measured in Phase 0; expected around 1B tokens each). This guarantees every run in the data-quality comparison sees its corpus less than once, so data quality is never confounded with repetition. Repetition is studied only in the dedicated repetition experiment. The exact shards used are listed in `data/manifest.json`.

### Splits

- Splits are made **by document**, before tokenizer training and before any training window is built.
- For each corpus, documents are shuffled with a fixed seed (1234). About 10 MB of UTF-8 text goes to **test**, about 10 MB to **validation**, and the rest to **train**.
- Exact-duplicate documents (by content hash) are removed across splits, so no train document appears in validation or test.
- **Contamination check.** Wikipedia text also appears in web crawls. Before final evaluation, every evaluation document is checked for 13-gram overlap with a large sample of the training data; documents with substantial overlap are removed and the count is reported.
- The test split is written once and never modified. Its SHA-256 hash is recorded in `data/manifest.json`.
- For the repetition experiment, smaller training sets are **nested random subsets** of the FineWeb-Edu train split (each smaller set contained in the larger).

---

## 4. Tokenizer

| Item | Declaration |
|---|---|
| Type | Byte-level BPE (Hugging Face `tokenizers`) |
| Baseline vocabulary | **8,192** tokens, including one `<\|endoftext\|>` document separator |
| Training text | An equal-sized sample of the **train** splits of FineWeb-Edu, FineWeb and DCLM. Validation and test text are never used |
| Storage | Token IDs stored as `uint16` |
| Budget | Tokenizer training is preliminary work (excluded, per the brief) |

Byte-level BPE encodes any text and decodes it back exactly, with no unknown tokens. This keeps bits-per-byte scoring exact.

**Rationale.** At our model sizes (roughly 8–120M parameters), GPT-2's 50,257-token vocabulary would place most parameters and most per-token compute in the embedding/output layer. A smaller vocabulary keeps the baseline's capacity and compute in the transformer layers.

The baseline vocabulary is fixed for the size, data and optimiser experiments. A separate vocabulary experiment compares 4,096 / 8,192 / 16,384 / 32,768 and GPT-2's 50,257 under the same budget.

---

## 5. Evaluation objective

**Objective:** predictive quality on educational English web text.

### Primary metric: bits per byte (BPB)

```
BPB = ( Σ −log p(x_i | x_<i) ) / ( B × ln 2 )
```

- The sum runs over every scored target token; `B` is the UTF-8 byte count of the text those targets represent.
- BPB is comparable across tokenizers because it is normalised by the same raw text, not by token count.

### Scoring protocol (identical for every run)

- Documents are concatenated with `<|endoftext|>` between them and cut into non-overlapping windows of the model's context length (baseline 512 tokens). Each target token is scored exactly once.
- `<|endoftext|>` targets contribute neither loss nor bytes.
- The first tokens of each window see less context than later ones. This is the same for all runs with the same context length; the vocabulary and context experiments report bytes per token alongside BPB.
- Scoring is deterministic: the same documents in the same order every time.

### Evaluation sets

| Set | Used for | When |
|---|---|---|
| Fixed validation subset (about 1 MB of FineWeb-Edu validation) | Learning curves | Every 5% of the budget (20 points per run), clock paused |
| Full FineWeb-Edu validation split | All tuning and design decisions | End of every run |
| **FineWeb-Edu test split** | **Primary reported result** | Once, at the end of the project, for frozen configurations only |
| FineWeb, DCLM and Wikipedia test splits | Data-quality experiment and generalisation check | Once, at the end, frozen configurations only |

### Decision rule for the data-quality experiment

A model trained on FineWeb-Edu will naturally score best on FineWeb-Edu text, so that test set alone cannot judge corpora fairly. For this experiment only, the headline result is the **equal-weight average BPB over the four test sets** (FineWeb-Edu, FineWeb, DCLM, Wikipedia). Each test set is also reported separately.

### Test-set lock

- `code/evaluate_test.py` refuses to run unless the configuration is listed in `configs/frozen_configs.yaml`.
- That file is committed before any test evaluation. The git history therefore shows the test set was never used for tuning.

### What the metric leaves out

BPB measures how well a model predicts this text distribution. It does not directly measure factual accuracy, reasoning or instruction following. Downstream benchmarks (for example HellaSwag) give near-chance scores at our model sizes and would not separate configurations; at most one is reported as a sanity check. Generated text samples (fixed prompts, temperature and length) are shown only as illustrations.

---

## 6. Checkpoint, seed and run policies

### Evaluated checkpoint

- The **final model at the end of the 60-minute budget** is the evaluated model for every run.
- The best-validation checkpoint is **not** selected. Its loss is logged as a diagnostic only (for example, the gap between best and final validation loss in the repetition experiment).

### Seeds

- A seed controls weight initialisation and training-data order. GPU kernels are not bitwise deterministic, so a seed controls these two things, not exact reproducibility.
- Seeds used: **0–4**, reused across configurations (seed 0 first, then 1, 2, …).

| Stage | Seeds per configuration |
|---|---|
| Noise calibration (baseline configuration, at the start of the size experiment) | 5 |
| Exploratory sweeps | 1 (seed 0) |
| Close calls (see rule below) | 3 |
| Final comparison: baseline recipe vs best recipe | 3 each, reported as mean ± std with every individual value |

**When a difference counts as real.** Let σ be the seed-to-seed standard deviation of final validation BPB from the 5-seed noise calibration. Comparing two configurations with *n* seeds each, the standard error of the difference is σ·√(2/n). A difference is called **real only if it exceeds twice this standard error**: 2.8σ with one seed each, 1.6σ with three seeds each. Smaller differences are reported as **inconclusive**, and a single-seed difference below 2.8σ triggers the close-call reruns. Five seeds (rather than three) are used for calibration because a standard deviation estimated from three runs is itself very uncertain.

### Run replacement

- A run is replaced **only** for an infrastructure failure: a crash, or throughput more than 10% below that configuration's throughput measured in its Phase 0–1 pilots.
- The failed run stays in the ledger and is re-run with the same seed.
- A run is **never** re-run because its result looks bad.

### Logged for every run

Configuration file, seed, git commit, Modal account (workspace), GPU name (`nvidia-smi`), library versions, exact parameter counts (total, embedding, non-embedding), counted training time, wall-clock time, tokens processed, unique training tokens and passes over the data, steps, tokens per second, peak memory, learning-rate history, validation curve, final validation BPB, and cost.

---

## 7. Codebase

- Based on **nanoGPT** (Karpathy), pinned to commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`.
- Project modifications, each documented in `readme.md`:
  - time-based budget enforcement and time-based cosine schedule (§1);
  - complete checkpoint state, including cumulative budget and RNG (§1);
  - deterministic BPB evaluator for validation and test (§5);
  - separate final and best-validation checkpoints (§6);
  - per-run JSON results and a compute ledger (§6);
  - Modal launch scripts.
- The `from_pretrained` path for GPT-2 weights is never called.

---

## 8. Report and repository

- Report written in **LaTeX** using the professor's Overleaf template. Source in `report/report.tex` with bibliography and figures.
- Every number in the report comes from files in `results/`; figures are generated by scripts in `code/plots/`.
- Repository layout:

```
readme.md          setup, data preparation, how to reproduce each result
authors.md         names and student numbers
DECLARATIONS.md    this file
report/            report.tex, references.bib, figures/
code/              training, evaluation, data preparation, Modal launch, plotting
configs/           one config file per run; frozen_configs.yaml
results/           one JSON per run; compute_ledger.csv
data/manifest.json shards used, split sizes, test-set hashes
```

---

## Amendments

| Date | Change | Reason |
|---|---|---|
| 2026-11-01 | `torch.compile` **on** for every run (its warm-up stays inside the 60 min) | E00 benchmark: +20 % steady throughput, 23-45 s warm-up (<1.3 % of the budget), net ~+19 % tokens per hour, less memory |
| 2026-11-01 | Batch: **32,768 tokens per optimiser step** (micro-batch 32 x 2 accumulation x context 512), same for every model size | E00: a larger micro-batch gave no speed gain; 32k tokens is well below the estimated critical batch size, so it is sample-efficient; one batch for all sizes keeps the size comparison clean. A batch-size experiment may follow in Phase 4 |
| 2026-11-01 | Model-size grid: **6 sizes** (6.9M, 11.4M, 17.5M, 25.6M, 35.9M, 64.6M total parameters), head dim 64, width/depth ratio ~43-53 | E00: measured throughput + Chinchilla allocation predict an optimum near 25-30M; grid is roughly log-spaced (ratio ~1.5) around it with brackets on both sides. If the best size is at an edge, one more size beyond that edge is run |
| 2026-11-01 | Full data download: **9,500 MB of training text per training corpus** (about 2.3-2.6B tokens each) | Declared rule: >= 1.5 x the tokens the fastest configuration (6.9M) processes in 60 min = 1.5 x 1.52B = 2.28B. No change to the rule itself |
| 2026-11-01 | LR tuning protocol made precise before any pilot ran (decision D07): (1) **edge rule**: if the best LR is at the grid edge, the next factor-2 value beyond it is run, and once more if it is again best (at most 2 extensions per configuration); (2) a pilot whose loss becomes NaN/inf is stopped and ranked worst; (3) **horizon check**: the 60-min check at the baseline size also runs the lower neighbour of the best LR when the 15-min runner-up is the higher neighbour, and the 15-min ranking counts as contradicted only if a different LR wins at 60 min by more than 2.8 sigma15 (sigma15 = seed sd of 3 pilots of the best baseline configuration); smaller flips are ties and the 15-min choices stand | The declared "one more value" could leave a size untuned, which would bias the size comparison; optimal LR is known to fall with longer training (Bjorck et al. 2024), which the top-2 check alone may miss; without a threshold a noise-level flip would shift every size |
| 2026-11-03 | **Peak learning rate per model size** (E01, 15-min pilots; close calls decided by the mean of 3 seeds): **N7 2e-3, N11 2e-3, N17 4e-3, N26 4e-3, N36 4e-3, N65 2e-3**. Horizon check at N26 (60 min): 4e-3 1.2058 vs 2e-3 1.2077 BPB, a tie well below the 0.0174 threshold, so no shift. All other settings as in the rows above (AdamW 0.9/0.95, weight decay 0.1, clip 1.0, 200 warm-up steps, cosine to 10 % of peak, 32,768 tokens/step, context 512, compile on) | Declared tuning protocol (section 1) as made precise on 2026-11-01; full record in the E01 experiment record (decisions D07-D14) |
| 2026-11-03 | **Amendment policy clarified** (replaces "no amendment is allowed after the first final run except to fix a bug" in the introduction): (1) the per-run compute budget (60 counted minutes on one L4) and the test-set lock never change; (2) the rules of a comparison (metric, decision rule, noise threshold, configurations compared) are fixed before that comparison's first run and never changed after it; (3) rules for experiments not yet started may be added later, recorded in `decisions.md` and in this table **before** their first run; (4) bug fixes are allowed and disclosed. Also: the 60-min N26 lr 4e-3 horizon run (E01-N26_lr4e-3_60min, seed 0) counts as the Phase 2 N26 seed-0 run, because it is a complete run under the declared budget with the Phase 2 configuration | Read literally, the original sentence would have frozen Phases 3-5 before any Phase 2 result, contradicting the assignment's emphasis on following up findings ("investigate unexpected results"; "follow-up on findings" in the 35 % experimental-design criterion). The topic page requires only the budget to be declared before experiments. What matters for fairness, not changing how a comparison is judged after seeing it, is kept. Reuse: the 4e-3 choice was made from 15-min results before the 60-min outcome, so the selection effect is negligible; disclosed in the report |
| 2026-11-03 | **Phase 2 (E02, model size) rules**, fixed before any new E02 run: 6 sizes (N7-N65) x 60 min at their E01 LRs, seed 0; N26 seeds 0-4 give sigma60. Best size: seed-0 ranking; every size within 2.8 sigma60 of the best (at most 3), and any size with an instability flag (> 5 % clipped steps or a loss jump > 0.1), gets seeds 1-2; the winner is the lowest 3-seed mean and is real only if it beats every other contender by > 1.6 x sigma pooled over all sizes with >= 2 seeds, otherwise an optimal range is reported. Edge sizes N4 / N92 if the smallest / largest wins; refinement size N49 (LR tuned by the declared protocol) if the fitted optimum lies between N36 and N65. Baseline size for later phases: the winner, or N26 if it is in the optimal range. Decisions use final FineWeb-Edu validation BPB only; FineWeb, DCLM and Wikipedia validation are also scored as information | Amendment policy (row above): a comparison's rules are recorded before its first run. Full reasoning in decisions D17-D18 and the E02 experiment record |
| 2026-11-03 | **E02 follow-up size N49** (11 x 576, 48.8M), declared after the E02 seed-0 results and before the N36/N65 close-call seeds: its LR is tuned by the declared protocol (4 rates x 15 min), then it is run at seed 0 and joins the best-size rule like any other size. Replaces the conditional trigger for N49, which relied on a parabola fit that had a coding error (fixed and disclosed) | Seed 0 placed the best sizes at 36M and 65M, the widest gap in the grid (x1.8); a point between them is needed to locate the optimum. Existing comparisons and their rules are unchanged (decision D19) |
| 2026-11-04 | **Phase 3a (E03, data quality) rules**, fixed before any E03 run: baseline size N49 (LR 2e-3) trained on FineWeb-Edu (E02 runs), raw FineWeb and DCLM, seeds 0-2 each. Decision metric as declared in section 5: equal-weight average of final validation BPB over FineWeb-Edu, FineWeb, DCLM and Wikipedia; a difference is real if it exceeds 1.6 x the pooled seed sd of that average. **Contamination check** before interpretation: a validation document is contaminated by a training corpus if at least 50 % of its token 13-grams occur in that corpus's train split; if more than 1 % of a validation set is contaminated, the average is also reported with the same documents removed for every arm. **Reported beside the decision, never replacing it:** the same comparison adjusted for machine speed (common-slope covariance on ln tokens processed; a pair whose verdict changes is reported as not robust); per-set results; a learning-rate check (15-min pilots at 2e-3 and 4e-3 on FineWeb and DCLM; if 4e-3 is lower on that corpus's own validation set by more than 2.8 x the 15-min seed sd, that arm also gets a 60-min run at 4e-3). **Conditional follow-up:** a 60/40 FineWeb-Edu/DCLM mixture (3 seeds) if those two are not separated by the decision rule or each is better on at least one validation set beyond noise. **Phases 4-6 train on FineWeb-Edu whatever E03 finds** | Sections 3 and 5 declare this experiment and its metric; this row fixes size, seeds, thresholds and the robustness checks before any run. Decisions D23, D25 |
| 2026-11-04 | **Phase 3b (E04, repetition) rules**, fixed before any E04 run: N49 on random nested subsets of the FineWeb-Edu train split of 290M/k tokens (k = 2, 4, 8, 16, 32; ~k passes in 60 min), seed 0, compared with the unique-data N49 runs (seeds 0-2): a run is worse if its FineWeb-Edu validation BPB exceeds their mean by more than 2 x sigma x sqrt(4/3). The smallest worse k and the k below it get two more seeds each; with 3 seeds a k is worse if it exceeds the reference by more than 1.6 x sigma; if the 3-seed verdicts move the boundary, it is moved by at most one more step (two more seeds there). Subsets are whole training windows chosen by a fixed permutation (key 1234), so they are uniform over the split and nested, as section 3 requires. Diagnostics (not decisive): machine-speed-adjusted deltas, train-validation gap, effective data and Muennighoff's R* | The existing first-N-tokens option would have drawn every subset from the first source file (the file the validation sample comes from), not a random subset. The boundary rule was made robust to non-monotonic results before any run. Decisions D23, D25 |
