# Project overview

An overview of the project: the question, how a run works, what each phase did and found, what did
not work, and what remains uncertain. Numbers are FineWeb-Edu validation bits per byte (BPB, lower is better) unless
marked "test". Every number links back to a write-up in [experiments/](experiments/) and from there to the result files.

---

## 1. The question

**If you only get one hour on one GPU to train a language model from scratch, how should you spend that hour to get the
best model?**

"Best" means the model that predicts unseen text best, measured in bits per byte. The tension: a big model learns more
from each piece of text but is slow, so it reads less text in the hour; a small model reads a lot of text but has less
capacity to learn from it. The dataset, tokenizer, optimiser and model shape all change that trade-off, so we tested them
one at a time and then combined the changes that helped.

## 2. The declared setup

Committed to git on 1 Nov 2026, before the first experiment, with every later change recorded as a dated
amendment: [DECLARATIONS.md](DECLARATIONS.md).

| | |
|---|---|
| Budget | 60 minutes of counted training time per run on one NVIDIA L4 (bf16), on Modal. Model compilation counts; evaluation and checkpoint pauses do not. Preliminary 15-minute learning-rate pilots and tokenizer training are excluded |
| Initialisation | Random weights only; nanoGPT's GPT architecture; our own BPE tokenizer |
| Data | FineWeb-Edu (main), FineWeb and DCLM (training alternatives), Wikipedia (evaluation only); documents split into train / validation / test once, with duplicates removed across splits |
| Metric | Bits per byte on held-out text: per byte, so it is comparable across tokenizers. FineWeb-Edu validation is the headline; four validation sets for the data experiment |
| Noise | Each comparison's threshold for a "real" difference is derived from measured seed-to-seed noise and fixed before its runs; smaller differences are reported as inconclusive |
| Test set | Scored once, at the end, only for models listed in advance (`configs/frozen_configs.yaml`) |

---

## 3. What one training run does

Every run (whatever the experiment) goes through the same steps:

1. **Start a computer in the cloud** (Modal) with one L4 GPU.
2. **Load the training text** (already turned into numbers, see "token" below) into memory.
3. **Create a brand-new model with random weights** (the rules forbid starting from a trained model).
4. **Start the 60-minute stopwatch** and train:
   - show the model a batch of text, it guesses each next word-piece,
   - measure how wrong it was, nudge its weights to be less wrong,
   - repeat thousands of times.
   - Every 3 minutes (5 % of the budget) the stopwatch pauses and we check the score on some
     held-out text, giving a learning curve. Pausing means checking does not eat the budget.
   - Every 10 minutes it saves a checkpoint, so if the cloud machine dies the run continues from
     there with the time already used still counted.
5. **Stop at 60 minutes.**
6. **Score the final model** on held-out validation text (bits per byte).
7. **Save everything**: the score, the learning curve, how much text it read, how fast it ran.
   On the launching computer this becomes `results/runs/<run name>.json` plus one row in
   `results/compute_ledger.csv`.

The only things that change between runs are the settings in the run's config file
(`configs/...yaml`): model size, data, learning rate, and so on. Comparing runs that differ in
one setting tells us what that setting does.

---

## 4. The project as a map

```
PHASE 0  Set up and check everything (E00)              data, tokenizer, training code, tests, speed benchmarks
   v
PHASE 1  Tune the learning rate per model size (E01)    15-min pilots + a 60-min check
   v
PHASE 2  Which model size is best in 60 minutes? (E02)  7 sizes; N49 becomes the baseline
   v
PHASE 3  Data: quality (E03) and repetition (E04)
   v
PHASE 4  Training choices: Muon optimiser (E05), model shape (E06)
   v
PHASE 5  Vocabulary size (E07)
   v
PHASE 6  Combine what helped (E08); score the test set once
```

Phases 0-1 are preparation. Phases 2-6 are the comparisons reported as results. In total the project used 270 Modal
jobs and about $79 of compute across four group members' workspaces (`results/compute_ledger.csv`).

---

## 5. What each phase found

**Phase 0 (E00, 1 Nov).** We downloaded 2.45-2.57 billion tokens per training corpus, enough that no run in the main
experiments reads the same text twice, trained an 8,192-token tokenizer, and adapted nanoGPT (time-based clock and
learning-rate schedule, checkpoint-and-resume, bits-per-byte evaluation). Short benchmarks set the size grid (7M-65M
parameters) and showed `torch.compile` is worth its warm-up (+20 % speed). A deliberate kill-and-resume test on Modal
found a bug that would have crashed every preempted run. [E00](experiments/E00_pipeline_and_benchmarks.md)

**Phase 1 (E01, 1-3 Nov).** Four learning rates per size in 15-minute runs, close calls repeated with more seeds, and a
60-minute check that a rate picked in 15 minutes still wins over the hour. **2e-3 or 4e-3 is best for every size**,
and the two are practically tied; larger models break at high rates. Identical runs differ by about 0.006 BPB, partly
because cloud GPUs run at slightly different speeds. [E01](experiments/E01_learning_rate_tuning.md)

**Phase 2 (E02, 3-4 Nov).** Seven sizes, each trained for one hour. **The three largest (36M, 49M, 65M) tie within
0.0015 BPB** and read only 4-11 tokens per parameter, far from Chinchilla's ~20; the 26M model, which reads ~20 per
parameter in our hour, is clearly worse. A post-hoc fit of a Chinchilla-style loss law to our own runs puts the
equal-compute optimum at ~56M parameters and ~5 tokens per parameter, so the gap is not explained by time vs FLOPs alone.
N49 (11 layers x 576) became the baseline for everything after. [E02](experiments/E02_model_size.md)

**Phase 3a: data quality (E03, 4-5 Nov).** Judged on the average over four validation sets (a model always looks best on
its own kind of text): **DCLM beats FineWeb-Edu and raw FineWeb by ~0.028**; FineWeb-Edu and FineWeb tie. A 60/40 mix of
FineWeb-Edu and DCLM beats all three on the average, at a cost of +0.022 on FineWeb-Edu itself. The ranking survives
removing contaminated validation documents and adjusting for machine speed. [E03](experiments/E03_data_quality.md)

**Phase 3b: repeated data (E04, 4-5 Nov).** With only a small slice of data, read 2-32 times in the hour: **2 passes are
as good as fresh data, 4 passes are measurably worse**, and at 32 passes the model memorises its slice (worse than an
untrained model on new text). Follow-ups: on a limited slice the largest model was still best (opposite of our
prediction); re-cutting the training windows every pass removes about a quarter of the damage. [E04](experiments/E04_repetition.md)

**Phase 4a: optimiser (E05, 5 Nov).** Muon, tuned with the same protocol as AdamW, **beats AdamW by 0.026 BPB in the same
hour**, on every validation set. Its extra computation costs only ~0.002 of that gain. [E05](experiments/E05_muon.md)

**Phase 4b: shape (E06, 5 Nov).** At the same 49M parameters, **6 wide layers (6 x 768) beat 11 x 576 by 0.018** because
they train 16 % faster on the L4 (tied at equal steps); 19 narrow layers (19 x 448) are worse by 0.021. [E06](experiments/E06_depth_width.md)

**Phase 5: vocabulary (E07, 5-6 Nov).** **16,384 and 32,768-token vocabularies both beat 8,192 by ~0.016**; 4,096 is
probably worse. About a third of the gain comes from the 512-token scoring window holding more text at larger
vocabularies. [E07](experiments/E07_vocabulary.md)

**Phase 6: the combined recipe (E08, 6 Nov).** Two recipes, 3 seeds each: **RB = Muon + 16k (declared final) -0.041**,
**RA = Muon + 16k + 6 x 768 -0.050** vs the baseline. Muon and the vocabulary gains add up almost exactly; the wide
shape's gain halves inside the recipe. RA vs RB (-0.009) was just inside the declared threshold, so RB is the final
recipe by the tie-break fixed in advance. On the test set, scored once for 57 frozen models: baseline 1.1994, RB 1.1583,
RA 1.1494; every validation conclusion held (test = validation + 0.005 for every model). [E08](experiments/E08_final_recipe.md)

---

## 6. What did not work, and what we changed

| What happened | What we changed |
|---|---|
| The first benchmark launch failed: YAML read `1.0e9` as text, and Modal retried each crash | Strict config loader; every config checked on the laptop before any GPU starts |
| Resuming after a preemption crashed on the GPU (random-number states loaded onto the GPU); CPU tests could not see it | Fixed; the Modal kill-and-resume test now passes |
| Two pilots reused a warm container and skipped the compile warm-up, gaining time | Detected from the first-step time; rerun with `--fresh`; one run per container enforced |
| Code was launched while uncommitted | `launch` refuses uncommitted code; every result records its commit and code hash |
| The laptop's connection to Modal dropped during many long runs | Runs continue on Modal; `recover` fetches results; the ledger keeps a `failed` row for each drop |
| Predictions that missed: optimal size (we expected 25-35M), smaller models on limited data (E04 F2), deep-narrow shape (E06), 32k vocabulary (E07) | Reported as misses; each has a candidate explanation in its write-up |

## 7. What remains uncertain

- One model scale and one GPU type: the wide-shape advantage is a property of how efficiently the L4 runs wide matrices.
- Some verdicts rest on one seed by the declared rule (6 x 768 alone, 32k, 4k); 4k was left as a close call.
- The best vocabulary is at the top of the tested range (32k); a larger one was not tried.
- Muon's own settings (momentum, Newton-Schulz steps) were left at their defaults; the embedding learning rate under Muon
  was checked only around the AdamW value.
- Bits per byte measures text prediction only, not factual accuracy or reasoning (DECLARATIONS section 5).

---

## 8. Glossary

| Word | Meaning |
|---|---|
| **Token** | A piece of text the model reads, usually a word or part of a word. "unbelievable" might be 3 tokens. Our text is about 3.9 bytes (characters) per token. |
| **Tokenizer** | The rule that cuts text into tokens. We trained our own: 8,192 tokens for the baseline, and 4,096 / 16,384 / 32,768 in E07. |
| **Parameters** | The numbers inside the model that training adjusts. "36M model" = 36 million of them. More = bigger capacity, slower. |
| **Train / validation / test** | Training text: the model learns from it. Validation: held-out text we use to compare and tune settings. Test: held-out text used **once** at the very end, so the final score is not biased by our choices. |
| **Bits per byte (BPB)** | Our score. How surprised the model is by unseen text, per character. Lower = better. Fair across tokenizers because it is per byte of text, not per token. |
| **Learning rate** | How big each nudge to the weights is. Too high: training blows up. Too low: learns slowly. Must be tuned per model size for a fair comparison. |
| **Throughput** | Tokens processed per second. Decides how much text a model reads in 60 min. |
| **torch.compile** | Optimises the model's code for the GPU. Faster steps, but takes time to set up at the start, and that time counts in our 60 minutes. |
| **Checkpoint** | A saved snapshot of the run, so it can resume after a crash without losing its place or resetting the clock. |
| **Pass / epoch** | Reading the whole training set once. 2 passes = every piece of text seen twice. |
| **Seed** | The random number that sets the starting weights and the order of the text. Repeating a run with different seeds shows how much results vary by luck. |
| **Muon** | An optimiser that updates each weight matrix as a whole, making every direction of the update equally strong (orthogonalised momentum). Compared with AdamW in E05 |
| **Validation vs test** | See "Train / validation / test" above: every decision in this project used validation; test was scored once, at the end |
| **Ledger** | `results/compute_ledger.csv`: one line per GPU job, to account for all compute used. |
| **Chinchilla rule** | From Reading 1: for the best model under fixed compute, use about 20 training tokens per parameter. We test whether that holds at our tiny scale. |

---

## 9. Which file to open for what

| I want to know... | Open |
|---|---|
| The fixed rules and every amendment | [DECLARATIONS.md](DECLARATIONS.md) |
| How to rerun anything | [REPRODUCE.md](REPRODUCE.md) |
| Details, numbers and problems of one experiment | [experiments/](experiments/) |
| Why each choice was made, in order (D01-D47) | [decisions.md](decisions.md) |
| What happened when, hour by hour | [experiment_log.md](experiment_log.md) |
| Every job, its time and its cost | [`results/compute_ledger.csv`](../results/compute_ledger.csv) |
