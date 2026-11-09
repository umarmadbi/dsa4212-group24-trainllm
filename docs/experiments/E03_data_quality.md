# E03: Which training corpus gives the best model in 60 minutes? (Phase 3a, data quality)


## Goal
Same model (N49), same hour, same tokenizer; only the text it learns from changes. FineWeb-Edu is web text
filtered for "educational" pages; DCLM is web text filtered by a different quality classifier; raw FineWeb is the
web text before either filter. Does the filter matter when the budget is one hour, and which kind of quality
helps on text the model was not trained on? A model always looks best on its own kind of text, so the judge is
the **average over four validation sets** (FineWeb-Edu, FineWeb, DCLM, Wikipedia), declared on 30 Oct.

Two extra checks make the comparison fair (added in D25, before any run): (1) the computers we rent are not all
equally fast, and a faster one reads more text in the hour, so we also compare the corpora **as if every run had
read the same number of tokens**; (2) the learning rate was tuned on FineWeb-Edu only, so four short pilots check
it is not clearly wrong for the other two corpora.

## Question
For N49 under the 60-min budget, which training corpus (FineWeb-Edu, FineWeb, DCLM) gives the lowest equal-weight
average of final validation BPB over the four validation sets?

## Prediction (written before any E03 run)
Each model is best on its own corpus's validation set. FineWeb-Edu is best on Wikipedia (encyclopedic style).
Differences in the 4-set average are small (within ~0.01 BPB); FineWeb-Edu or DCLM lowest, raw FineWeb highest.
What would change our mind: raw FineWeb winning the average (breadth beating filtering at this scale).

## Design
- **Arms (3 seeds each, seeds 0-2):** FineWeb-Edu = E02-N49 s0-2 (already run, identical config except corpus);
  FineWeb = `configs/E03/N49_fineweb.yaml`; DCLM = `configs/E03/N49_dclm.yaml`. 6 new 60-min runs (~$6).
- **LR check pilots (R6, D25):** `configs/E03/N49_{fineweb,dclm}_lr{2e-3,4e-3}_15min.yaml`, seed 0, 15 min,
  scored on all four validation sets. 4 pilots (~$1.1). The FineWeb-Edu side already exists (E01 N49 pilots,
  3 seeds each: 2e-3 1.3508, 4e-3 1.3593).
- **Held fixed:** N49 (11 x 576), LR 2e-3, 32,768 tokens/step, context 512, AdamW, schedule, budget, compile, bf16,
  8k tokenizer (trained on an equal mix of all three train corpora, so it does not favour one corpus by design).
- **Workspaces (each arm split over two workspaces, so a workspace cannot masquerade as a corpus effect):**
  giraffes573: FineWeb s0,s1 + DCLM s2 + the 4 R6 pilots; jinkyungjeon: DCLM s0,s1 + FineWeb s2 (the FineWeb-Edu
  arm ran there too).

## Decision rules (fixed D23; R3 made precise and R5-R8 added in D25, before any E03 run)
- **R1 Decision metric:** AVG4 = mean of final val BPB on FineWeb-Edu, FineWeb, DCLM, Wikipedia (each run). Arm value
  = mean over seeds 0-2. A difference between two arms is **real if > 1.6 sigma_avg4**, where sigma_avg4 = pooled
  seed sd of AVG4 over every E02 size with >= 2 fully scored seeds (0.0055, 9 df; re-derived from the run files in
  D25) and the two new arms (pooled at decision time, ~13 df). Smaller differences: inconclusive (reported, never
  called wins). Runs with an infrastructure flag are replaced before R1 is read.
- **R2 Per-set results** are reported for every arm (the "own-corpus advantage" is itself a finding) but never used
  for the decision.
- **R3 Contamination sensitivity (made precise, D25):** before E03 is interpreted, every validation document of the
  four sets is checked for token 13-gram overlap (our 8k BPE tokens, `<|endoftext|>` excluded) with the **full train
  split** of each training corpus (`code/contamination.py`). A document counts as contaminated by a corpus if
  **>= 50 % of its 13-grams occur in that corpus's train split** (documents shorter than 13 tokens are skipped);
  counts at >= 10 % and >= 80 % and the full distribution are reported too. If any validation set has > 1 % of its
  documents contaminated by any training corpus, AVG4 is recomputed for **every arm with the same documents
  removed** (the union over the three training corpora, so the comparison stays paired; rescoring the saved final
  models with `code/rescore.py`) and reported beside R1; R1 itself is unchanged.
- **R4** Infrastructure checks as in E02 (slow machine vs N49 pilot speed, warm compile, reused container, dirty
  code, instability); replacement only for infrastructure faults.
- **R5 Machine-speed sensitivity (D25):** the same pairwise comparison after adjusting each arm's mean AVG4 to equal
  tokens processed: ANCOVA with a common slope on ln(tokens processed), slope pooled within configurations over
  E02 (sizes with >= 2 seeds) and the new arms (E02 alone: -0.131 per unit ln tokens, residual sd 0.0047). Threshold
  1.6 x residual sd. Within one configuration, tokens processed differ only because of machine speed (the corpus
  does not change tokens/s), so this is a pure speed covariate. **Reported beside R1; if it changes R1's verdict for
  a pair, that pair is reported as "not robust to machine speed".** R1 remains the declared decision.
- **R6 LR fairness (D25):** for FineWeb and for DCLM, the 15-min pilots at 2e-3 and 4e-3 (seed 0) are compared on that
  corpus's own validation set (the set one would tune on). **Inconsistent** if 4e-3 is lower than 2e-3 by more than
  2.8 sigma15 = 0.0188 (the E01 single-seed threshold). If inconsistent for a corpus, that arm is also run at 4e-3
  for 60 min (seed 0) and its AVG4 reported as a sensitivity; R1 still uses the declared 2e-3 arms.
- **R7 Descriptive (D25, never decisive):** 3 x 4 table (arm x validation set) with the pooled per-set seed sd;
  Wikipedia alone (the only set no arm trains on); each arm's in-domain advantage (its own-set BPB minus the mean of
  the other arms on that set); text read in GB.
- **R8 Mixture follow-up trigger (D25, fixed before any E03 result; FIRED, D26):** a FineWeb-Edu/DCLM mixture (60/40 by tokens,
  the option named in the plan; SmolLM2 found the two complementary) is run, seeds 0-2, judged by R1 against both
  parents, **if** FineWeb-Edu vs DCLM is inconclusive under R1, **or** each is lower than the other on at least one
  validation set by more than 1.6 x that set's pooled seed sd (complementary profiles). Needs a corpus-mix option in
  `train.py` (built and tested only if the trigger fires; its rules go into this record before its runs).
  **Mixture rules (D26, fixed before its runs):** `configs/E03/N49_mix_fwe60_dclm40.yaml`, 60/40 FineWeb-Edu/DCLM by
  tokens (every window slot goes to the corpus furthest below its share), seeds 0-2. Each corpus contributes a 1B-token
  random pool of windows (nested-subset key 1234), loaded before the clock: two full splits would not fit the 16 GB
  container, and 1B is 6-9x what an hour reads, so nothing repeats (as in the single-corpus arms). Judged by R1 against
  both parents (pooled sigma including the mix arm), R5 alongside; secondary: its FineWeb-Edu val vs the FineWeb-Edu
  arm (1.6 sigma60 = 0.0091), i.e. "how much of the declared objective does the mix give up?". Re-scored without
  contaminated documents like every arm (R3). Workspaces: giraffes573 s0, s1; jinkyungjeon s2.
  **Prediction (D26, before its runs):** AVG4 within noise of DCLM and clearly better than FineWeb-Edu; FineWeb-Edu val
  ~0.02-0.03 worse than the FineWeb-Edu arm but much better than the DCLM arm (+0.08).
- **R9 Which corpus Phases 4-6 use (D25, fixed now):** **FineWeb-Edu, whatever E03 finds.** The declared objective
  (DECLARATIONS section 5) is predictive quality on educational web text, judged on the FineWeb-Edu test split; E03
  answers a separate question (generalisation across text types). Choosing the later corpus after seeing E03 would
  be a post-hoc change of the main comparison's training data.

## Gaps and confounders checked
- **Repetition:** none; each arm reads ~290M of its ~2.5B training tokens (0.12 passes).
- **Machine speed (D25):** 84 % of N26's seed variance follows machine speed (E02, P8). The FineWeb-Edu arm ran on
  slowish L4s (-2.4 to -3.8 % vs pilots). At the E02 slope, a 4 % difference in tokens is ~0.005 AVG4, over half
  the R1 threshold (0.0089), so speed could create or hide a corpus effect. Handled by R5 (and by splitting each new
  arm over two workspaces).
- **LR tuned on FineWeb-Edu only:** E01 showed 2e-3 vs 4e-3 within noise at every size, so a corpus-specific optimum
  is unlikely to be far off; R6 now checks the direction on each corpus (D25). Weight decay, warm-up and batch were
  not re-tuned per corpus: limitation.
- **Text per token differs:** bytes per token are 3.87 (FineWeb-Edu), 3.76 (FineWeb), 3.63 (DCLM) with our
  tokenizer, so in the same hour the DCLM model reads ~6 % less text. That is part of what "training on DCLM in
  60 minutes" means; the summary reports GB of text read per arm.
- **Own-corpus advantage is not perfectly symmetric (D25):** each arm has one in-domain set, but the four sets pair
  up by style: FineWeb-Edu and Wikipedia are both explanatory/encyclopedic, FineWeb and DCLM are both general web.
  The FineWeb-Edu arm may therefore get a near-domain bonus on Wikipedia, and FineWeb and DCLM on each other. This is
  a property of the declared metric, not a bug; R7 makes it visible (Wikipedia alone; in-domain advantage).
- **FineWeb-Edu is a subset of FineWeb:** the FineWeb validation set contains some educational pages, so the FineWeb
  arm's training distribution partly covers FineWeb-Edu text, but not the reverse. Stated.
- **Evaluation leakage:** exact duplicates of any evaluation document were removed from every train split, across
  corpora (D02; manifest: 0-5 per corpus). Near-duplicates handled by R3 (threshold fixed now, D25).
- **Validation sample from one source file per corpus (D02 open item):** FineWeb/FineWeb-Edu val and test come from
  source file 000 of each sample, DCLM from its first shard. If a source file is not representative (e.g. grouped by
  crawl dump), every arm is judged on the same slightly narrow sample. Same for all arms; limitation.
- **Noise:** sigma_avg4 is estimated mostly from FineWeb-Edu-trained models; the new arms' own sds are reported and
  enter the pooled estimate.
- **Mixed commits:** the FineWeb-Edu arm ran from 0b7dde4; new arms run from the D23/D25 commit, whose only
  training-code change (random subsets) is inactive for E03. `summarize.py E03` prints commit per run.
- **Not tested:** mixtures unless R8 fires; LR beyond the 2e-3/4e-3 check; a corpus effect at other model sizes.

## Run order
1. Commit D23 + D25 (code, configs, DECLARATIONS). `modal profile current` before each launch.
2. giraffes573: `configs/E03/N49_fineweb.yaml` seeds 0,1; `configs/E03/N49_dclm.yaml` seed 2; the four
   `*_15min.yaml` pilots seed 0 (`--kind pilot`).
3. jinkyungjeon: `configs/E03/N49_dclm.yaml` seeds 0,1; `configs/E03/N49_fineweb.yaml` seed 2.
4. While they run (after committing D25b): `modal run --detach modal_app.py::contamination` on any one workspace
   (then `--fetch-only` from the same workspace if the connection drops). If R3 requires it, rescore every arm's
   runs from its own workspace: `modal run modal_app.py::rescore --run-ids "..." --exclude contamination`.
5. `python code/summarize.py E03`: replace flagged runs first, then R1, R5, R6, R7, R8; then R3.

## Runs
Commit 4fce312 (D23 + D25), launched 11:54-12:00; all launch terminals lost their connection (14 'failed' ledger rows,
to annotate); every run completed and was recovered ~15:00. All clean: 1 attempt, cold compile, one run per container,
no instability flag.

| Run | Workspace | Start (SGT) | FWE val | FineWeb val | DCLM val | Wikipedia val | AVG4 | Tokens | Speed vs pilots |
|---|---|---|---|---|---|---|---|---|---|
| FineWeb s0 | giraffes573 | 12:04 | 1.2670 | 1.3315 | 1.5086 | 1.4727 | 1.3950 | 297M | -0.5 % |
| FineWeb s1 | giraffes573 | 12:15 | 1.2734 | 1.3377 | 1.5197 | 1.4822 | 1.4033 | 293M | -2.5 % |
| FineWeb s2 | jinkyungjeon | 12:12 | 1.2637 | 1.3281 | 1.5102 | 1.4682 | 1.3925 | 284M | -4.9 % |
| DCLM s0 | jinkyungjeon | 12:32 | 1.2698 | 1.3763 | 1.3562 | 1.4433 | 1.3614 | 284M | -5.1 % |
| DCLM s1 | jinkyungjeon | 13:04 | 1.2795 | 1.3844 | 1.3651 | 1.4514 | 1.3701 | 284M | -5.5 % |
| DCLM s2 | giraffes573 | 12:18 | 1.2795 | 1.3848 | 1.3652 | 1.4524 | 1.3705 | 294M | -2.1 % |

R6 pilots (giraffes573, 15 min, seed 0, 12:29-13:22): own-val BPB FineWeb 2e-3 1.4889 / 4e-3 1.4872; DCLM 2e-3 1.5267 /
4e-3 1.5255.
Contamination check (giraffes573, CPU, 283-419 s per training corpus): `results/contamination.json`.

## Results
| Trained on (3 seeds) | FWE val | FineWeb val | DCLM val | Wikipedia val | **AVG4** | sd | GB text read |
|---|---|---|---|---|---|---|---|
| FineWeb-Edu (E02-N49) | **1.1945** | 1.3850 | 1.5404 | 1.4616 | 1.3954 | 0.0015 | 1.12 |
| raw FineWeb | 1.2680 | **1.3324** | 1.5128 | 1.4744 | 1.3969 | 0.0056 | 1.09 |
| **DCLM** | 1.2763 | 1.3819 | **1.3622** | **1.4490** | **1.3673** | 0.0051 | 1.04 |

- **R1 (decision):** pooled sigma_avg4 0.0055 (13 df), threshold 0.0088. **DCLM better than FineWeb-Edu by 0.0280 and
  than FineWeb by 0.0296: REAL.** FineWeb-Edu vs FineWeb 0.0016: inconclusive.
- **R5 (speed):** slope -0.093 per unit ln tokens, residual sd 0.0050; adjusted DCLM 1.3665, FWE 1.3956, FW 1.3974;
  same verdicts. The DCLM arm had the slowest machines, so speed worked against it.
- **R6 (LR):** consistent on both corpora (4e-3 better by only 0.0017 and 0.0012 vs threshold 0.0188).
- **R3 (contamination):** union over training corpora at >= 50 %: FWE val 67/1981 (3.38 %), FineWeb 46/3057 (1.50 %),
  DCLM 24/1904 (1.26 %), Wikipedia 8/3173 (0.25 %) -> **re-score required; done 17:20:** without those docs (same 145 docs removed for every arm) DCLM is better by
  0.0290 (vs FWE) and 0.0298 (vs FineWeb), REAL; FWE vs FineWeb 0.0007 inconclusive. Conclusion unchanged. Own-corpus near-duplicates:
  FWE 2.83 %, FineWeb 1.14 %, DCLM 0.42 % of their val docs: contamination favours FineWeb-Edu, not DCLM. Mean 13-gram
  hit rate of a val set in its own corpus's train split: FWE 5.5 %, FineWeb 3.6 %, DCLM 3.3 %.
- **R7:** DCLM lowest on Wikipedia (by 0.0126 vs FWE, per-set noise scale 0.0117) and equal to FWE on raw FineWeb val.
  In-domain advantage: DCLM -0.164, FWE -0.078, FineWeb -0.051.
- **R8:** fired -> mixture arm (rules above).

### Mixture (R8, D27)
| Run | Workspace | Start | FWE val | FineWeb val | DCLM val | Wikipedia val | AVG4 | Tokens | Speed |
|---|---|---|---|---|---|---|---|---|---|
| mix s0 | giraffes573 | 4 Nov 17:2x | 1.2134 | 1.3682 | 1.3781 | 1.4099 | 1.3424 | 306M | +2.2 % |
| mix s1 | giraffes573 | 4 Nov 17:2x | 1.2167 | 1.3713 | 1.3814 | 1.4144 | 1.3460 | 286M | -4.8 % |
| mix s2 | jinkyungjeon | 4 Nov 17:2x | 1.2200 | 1.3743 | 1.3837 | 1.4206 | 1.3496 | 292M | -2.2 % |
| **mix mean** | | | 1.2167 | 1.3713 | 1.3811 | **1.4150** | **1.3460** | | |

- R1: mix better than DCLM by 0.0213, FWE by 0.0494, FineWeb by 0.0509 (threshold 0.0085, pooled sigma 0.0053, 15 df): REAL.
  R5 (speed): 0.0191 over DCLM, REAL. R3 (no contaminated docs): 0.0208 over DCLM, REAL.
- Cost on the declared objective: FWE val +0.0222 vs the FWE arm (beyond 0.0091). Best of all arms on Wikipedia (by 0.034)
  and better than both parents on raw FineWeb val.
- Prediction partly missed (we expected AVG4 within noise of DCLM); FWE-val cost as predicted.

## Findings
1. **Under a 60-min budget, a 60/40 FineWeb-Edu/DCLM mix gives the best average over four kinds of text** (AVG4 1.346),
   0.021 better than DCLM alone and 0.049 better than FineWeb-Edu alone, and is best of all on Wikipedia, which no arm
   trained on (by 0.034). It gives up 0.022 on FineWeb-Edu text. **Of the single corpora, DCLM is best on average** (0.028
   ahead of FineWeb-Edu, ~5 sigma) and also beats FineWeb-Edu on Wikipedia.
2. **FineWeb-Edu is best only on FineWeb-Edu-like text, but there by a lot** (0.07-0.08 BPB). Since the declared
   objective is FineWeb-Edu, Phases 4-6 stay on FineWeb-Edu (R9, declared before the results).
3. **The educational filter does not beat raw web text on average** (FWE vs FineWeb inconclusive): what it gains on
   educational text it loses on everything else.
4. Not explained by machine speed, the learning rate, or contamination (all checked by declared rules).

## Surprises, failures and what we changed
- **Prediction missed:** we expected differences within ~0.01 with FineWeb-Edu or DCLM lowest; DCLM wins by ~0.03.
- Contamination fetch failed once on the laptop (numpy missing in the launch-only environment; nothing had started).
  Fixed (lazy import, D26).
- R8's "complementary profiles" condition fires almost automatically when each corpus wins on its own text; it was
  declared that way, so the mixture is run, and this is noted.
