# E06: Deep-and-narrow or shallow-and-wide at the same size? (Phase 4b, shape)

## Summary
- **Question:** with the parameter count fixed at ~49M and 60 counted minutes on one L4, is a shallow-wide model (6 layers
  x 768) or a deep-narrow one (19 x 448) better than N49's 11 x 576?
- **Answer:** **6 x 768 is better (1.1768, -0.018) and 19 x 448 is worse (1.2158, +0.021)** than N49's 3-seed mean 1.1945,
  each decided at seed 0 by the declared rule (threshold 0.0132). The order follows the width-to-depth ratio: 128 > 52 > 24.
- **Why:** the wide model runs 16 % faster on the L4 (MFU 0.26 vs 0.22) and reads 348M tokens instead of 291M; at equal
  steps it ties the baseline (-0.002). Its whole advantage is throughput, i.e. hardware-specific. The deep model loses
  both per step (+0.015) and in speed (-12 %).
- **Held on every check:** machine-speed adjustment (-0.015 / +0.026), all four validation sets, and the test set, scored
  once at the end (-0.018 / +0.021).
- **Tuning:** each shape got the full LR protocol (4 values, close calls settled by 3 seeds); both ended at 2e-3, like
  the baseline.
- **Consequence:** 6 x 768 entered recipe RA in Phase 6 (E08). With Muon and the 16k vocabulary its gain halved (-0.009),
  just inside the noise threshold, so the declared final recipe RB keeps 11 x 576.
- **Prediction missed:** we expected no shape to be clearly better and the wide model to be only slightly faster.
- **Figure:** no dedicated figure; width's contribution inside the recipe is in `report/figures/recipe_attribution.pdf`
  (`code/plots/fig_recipe_attribution.py`).
- **Cost:** 16 fifteen-minute LR pilots and 2 sixty-minute runs; the baseline reuses E02-N49. 6.0 counted GPU-hours, ledger
  $5.17, 5 Nov 2026.

## Goal
With the number of parameters fixed (~49M), a model can be few thick layers or many thin ones. Kaplan et al. found the
shape matters little; MobileLLM found deep-and-thin better for small models. Under a TIME budget there is a second
effect: thick layers use the GPU better (big matrix multiplications), thin-and-deep ones run more, smaller operations
in sequence, so they may read less text in the hour. Which effect wins on our L4?

## Question
At ~49M parameters and 60 min, does a shallow-wide (6 x 768) or deep-narrow (19 x 448) model beat N49 (11 x 576)?

## Prediction (written before any E06 run)
Throughput: 6 x 768 slightly faster (higher MFU), 19 x 448 slower (~5-10 %). BPB: all three within ~0.01; if anything
the baseline or the deep model slightly better per token, cancelled by the deep model's lower speed. Most likely
result: no shape clearly better than 11 x 576 (Kaplan's "shape matters little" holds under a time budget).

**Outcome: missed.** The wide model was much faster than "slightly" (+16 % tokens/s); the deep model was 12 % slower
(predicted 5-10 %). At 60 minutes the wide model is better by 0.018 and the deep model worse by 0.021. The per-token part
held for the wide model (a tie at equal steps) but not for the deep model (worse per step). Kept as written.

## Design
| Shape | Layers x width | Heads (dim 64) | Params (total / non-emb) | Width / depth |
|---|---|---|---|---|
| W6x768 (wide) | 6 x 768 | 12 | 49.2M / 42.5M | 128 |
| N49 (baseline, E02) | 11 x 576 | 9 | 48.8M / 43.8M | 52 |
| D19x448 (deep) | 19 x 448 | 7 | 49.7M / 45.8M | 24 |
(Parameter counts re-checked in D30 by building the models.)
- **LR (D30: the full declared protocol, as every size had in E01):** 15-min pilots at 5e-4, 1e-3, 2e-3, 4e-3, seed 0
  (`configs/E06/*_lr*_15min.yaml`), edge rule (next factor-2 value beyond, at most twice) and close-call rule (seeds 1, 2
  of the top two, lower 3-seed mean wins). D28 had only 2e-3 and 4e-3 with an E01 "width rule" default: with two values
  the best is always at an edge and nothing checks whether 1e-3 (wide) or 8e-3 (deep) would be better, and the shape
  comparison would then partly measure LR tuning. Extra cost ~$1.2. Other settings as E02-N49 (AdamW).
- Finals: `configs/E06/W6x768_lr<pick>_60min.yaml`, `D19x448_lr<pick>_60min.yaml` (one file per LR; no placeholder),
  60 min. Baseline = E02-N49 s0-2.
- Workspace: jinkyungjeon. Cost: 8 pilots (~$2.4), edge/close-call pilots 0-8 (~$0-2.4), 2 finals (~$2.4), up to 4
  close-call seeds (~$4.8).

## Decision rules (fixed D28; revised D30 before any run)
- **R1 LR per shape:** the declared protocol above (same code as E05 R1).
- **R2 Seed 0:** each shape vs the N49 3-seed mean; |diff| > 2 sigma60 sqrt(4/3) = 0.0132 -> decided at seed 0;
  otherwise seeds 1, 2 and the 3-seed means are compared at 1.6 sigma60 = 0.0091.
- **R3 Infrastructure (D30 fix):** each shape's finals against **its own pilots' speed** (> 10 % slower -> replaced, same
  seed; excluded from R2 until replaced). The D28 code ran no speed check at all for E06 (it passed an empty reference).
- **R4 Reported, never decisive:** tokens/s and MFU vs 11 x 576, tokens read; per step vs per time (learning curves; see
  E05 R4a); machine-speed sensitivity; AVG4 and the four sets; the same comparison at 15 min; LR sensitivity per shape.

## Gaps and confounders checked
- **Parameter counts differ by < 2 %** (48.8-49.7M); non-embedding counts differ more (42.5-45.8M) because the
  embedding scales with width. Both reported. The deep model has the most non-embedding parameters, which slightly
  favours it per token.
- **LR tuning** now equal to every other configuration (D30).
- **Warm-up 200 steps** is the same in steps; steps per hour differ by shape (reported).
- **Head count differs** (head dim fixed at 64). Inherent to changing width.
- **Init:** nanoGPT scales residual projections by 1/sqrt(2 x layers), so the deep model's residual branches start
  smaller; this is part of "the shape" as nanoGPT defines it (not changed).
- **Only two alternative shapes, one size:** Kaplan's flat region spans width/depth ~20-200; our shapes are 24, 52, 128,
  i.e. they cover it. Limitation: one parameter count.

## Runs
| Round | Runs | Workspace / commit | Launched (SGT) | Status |
|---|---|---|---|---|
| 1 | LR pilots 5e-4, 1e-3, 2e-3, 4e-3 x {6x768, 19x448}, s0, 15 min | jinkyungjeon / 7e1b431 | 5 Nov 10:35 (starts 10:40-12:03) | done, clean, recovered ~12:20 |
| 2 | close calls: 6x768 1e-3, 2e-3 and 19x448 2e-3, 4e-3 x s1, s2 (15 min) | jinkyungjeon / 7e1b431 | 5 Nov ~12:30 (starts 12:31-13:31) | done, clean (D33) |
| 3 | 60 min s0: W6x768 2e-3, D19x448 2e-3 | jinkyungjeon / 7e1b431 | 5 Nov 15:17 | done, clean (D34) |

## Results
**Round 1 (15 min, seed 0):** 6x768: 5e-4 1.3702, 1e-3 1.3272, **2e-3 1.3194**, 4e-3 1.3602 (close call 2e-3 vs 1e-3, gap 0.0078).
19x448: 5e-4 1.4861, 1e-3 1.4151, **2e-3 1.3851**, 4e-3 1.3925 (close call 2e-3 vs 4e-3, gap 0.0074). Speed vs 11x576: wide +15.8 %
(MFU 0.26 vs 0.23), deep -12.4 % (MFU 0.21). At 15 min (descriptive): wide -0.031, deep +0.034 vs N49's 1.3508. Details D32.

**Round 2:** picks 6x768 **2e-3** (1.3178 vs 1e-3 1.3259), 19x448 **2e-3** (1.3846 vs 4e-3 1.3933), 3-seed means. Details D33.
**Round 3 / decision (seed 0 vs N49 3-seed mean 1.1945, threshold 0.0132):** 6x768 **1.1768 (-0.0176) BETTER** (speed-adjusted -0.0148);
19x448 **1.2158 (+0.0213) WORSE** (speed-adjusted +0.0257). Equal steps: wide -0.0016 (tie), deep +0.015. Details D34.


### Speed and tokens (60 minutes; baseline = mean of 3 seeds)
| Shape | Tokens/s | MFU | Tokens read | Steps | Tokens per parameter | Non-embedding params | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| 6 x 768 | 96,801 | 0.257 | 348M | 10,612 | 7.1 | 42.5M | 4.5 GB |
| 11 x 576 (E02-N49) | 81,448 | 0.222 | 291M | 8,869 | 6.0 | 43.8M | 5.5 GB |
| 19 x 448 | 74,227 | 0.214 | 263M | 8,012 | 5.3 | 45.8M | 6.8 GB |

### Per step vs per time (R4a; learning-curve subset)
| Shape | At equal time (declared) | At equal steps | What the speed difference is worth |
|---|---|---|---|
| 6 x 768 | -0.018 | -0.002 (its own curve read at the baseline's 8,869 steps) | -0.016 |
| 19 x 448 | +0.023 | +0.015 (the baseline's curve read at 8,012 steps) | +0.008 |

The curve point read off at the other arm's step count is not fully annealed (learning rate 0.12-0.16 x peak instead of
0.10), which slightly favours the other arm in the equal-steps view.

### On the other validation sets
| Shape | FineWeb-Edu | FineWeb | DCLM | Wikipedia | Average of four |
|---|---|---|---|---|---|
| 6 x 768 (seed 0) | 1.1768 | 1.3682 | 1.5275 | 1.4473 | 1.3800 |
| 11 x 576 (3 seeds) | 1.1945 | 1.3850 | 1.5404 | 1.4616 | 1.3954 |
| 19 x 448 (seed 0) | 1.2158 | 1.4046 | 1.5587 | 1.4799 | 1.4147 |

The wide model is better on every set (-0.013 to -0.018), the deep model worse on every set (+0.018 to +0.021).

### Learning-rate sensitivity at 15 minutes (seed 0; 3-seed mean in brackets where a close call was run)
| LR | 5e-4 | 1e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|
| 6 x 768 | 1.3702 | 1.3272 (1.3259) | 1.3194 (1.3178) | 1.3602 |
| 11 x 576 (E01) | 1.4061 | 1.3774 | 1.3577 (1.3508) | 1.3710 (1.3593) |
| 19 x 448 | 1.4861 | 1.4151 | 1.3851 (1.3846) | 1.3925 (1.3933) |

All three shapes are best at 2e-3: at this size the shape does not change the learning rate. The deep model loses most at
low learning rates.

### Test set (scored once at the end, D47)
| Shape | Seeds | Validation | Test | Test without contaminated documents | Average of four, test | Test vs baseline |
|---|---|---|---|---|---|---|
| 6 x 768 | 1 | 1.1768 | 1.1814 | 1.1819 | 1.3787 | -0.0180 |
| 11 x 576 (E02-N49) | 3 | 1.1945 | 1.1994 | 1.1998 | 1.3947 | - |
| 19 x 448 | 1 | 1.2158 | 1.2205 | 1.2208 | 1.4143 | +0.0211 |

FineWeb-Edu test, the two E06 finals listed in `configs/frozen_configs.yaml`, scored once after all experiments; nothing
was decided from test. `python code/summarize.py TEST`; files `results/test/E06-*.json`.

## Findings
- At equal parameters and equal time on the L4, fewer wider layers win: 6x768 < 11x576 < 19x448 (width/depth 128 > 52 > 24).
- The wide model's advantage is throughput, not per-token learning: +16 % tokens/s (MFU 0.26 vs 0.22) and a tie at equal steps.
  Hardware-specific: it reflects how efficiently this GPU runs wide vs narrow matmuls.
- The deep model loses both per step and in speed; MobileLLM's deep-thin advantage does not appear at 49M under a time budget.
- Prediction ("no shape clearly better") missed. Limitation: the wide verdict rests on one seed (declared rule; 1.3x the threshold).
- Held on the test set: -0.018 and +0.021, the same as on validation.

### Outcome of every declared rule
| Rule | Fired? | Outcome |
|---|---|---|
| R1 LR per shape | Yes, both close calls | 6 x 768: 2e-3 vs 1e-3 (gap 0.0078) -> **2e-3** (1.3178 vs 1.3259); 19 x 448: 2e-3 vs 4e-3 (gap 0.0074) -> **2e-3** (1.3846 vs 1.3933). Best values interior, so the edge rule was not needed |
| R2 seed-0 decision | Yes, decided at seed 0 | 6 x 768 -0.0177 **BETTER**; 19 x 448 +0.0213 **WORSE** (threshold 0.0132); the close-call seeds were not needed |
| R3 infrastructure | Clean | Speed vs each shape's own pilots -0.8 % and +0.3 %; all 18 runs one attempt, cold compile, full budget; none replaced |
| R4 reported | Yes | Tables above: speed and MFU, per step vs per time, machine-speed adjustment (-0.015 / +0.026), four sets, 15-minute comparison (-0.033 / +0.034, 3 seeds each), LR sensitivity |

## Surprises, failures and what we changed
- **Prediction missed:** the wide model's speed advantage (+16 %) was far larger than expected, and it accounts for the
  whole gain.
- **The deep model was worse per step as well**, against MobileLLM's finding for small models.
- **The first design tested only 2 LRs per shape (D28).** The review before any run (D30) replaced it with the full
  4-value protocol: with two values the best is always at an edge, and 1e-3 for the wide model, which the old design would
  never have run, turned out to be its close call. The same review found that the E06 summary had no slow-machine check
  at all; fixed before any run.
- **The wide verdict rests on one seed at 1.3x the threshold.** The review before Phase 6 (D41) flagged it as a
  multiple-comparison risk; Phase 6 re-tested width with 3 seeds inside the recipe, where its gain halved (see below).

## How this relates to the readings
- **Kaplan et al. (2020):** at a fixed number of non-embedding parameters, loss depends only weakly on the shape over a
  wide range of width-to-depth ratios, compared at equal training tokens. Per step we agree for 6 x 768 vs 11 x 576 (a tie),
  but 19 x 448 (ratio 24, near the low end of that range) is worse per step. Under a time budget, shape matters mainly
  through throughput, which a per-token analysis does not see.
- **Liu et al. (2024), MobileLLM:** for sub-billion models, deeper and thinner beats wider at the same parameter count.
  We do not see this at 49M: the deep model is worse per step and slower. Their models are deeper (30 or more layers),
  include other changes (SwiGLU, grouped-query attention, embedding sharing) and are compared at equal training tokens, not
  equal time; our time budget penalises depth through throughput.
- **Topic page:** asks how parameters should be allocated across depth and width. Our answer, for this GPU and size: wider
  within the range tested, because wide matrix multiplications run more efficiently, not because wide layers learn more
  per token.

## Limitations and what remains uncertain
- **One seed per shape** (the declared seed-0 rule); the wide verdict is 1.3x the threshold.
- **One parameter count (~49M), two alternative shapes, AdamW only.** Shape x Muon was tested only inside the E08 recipe.
- **One GPU type, power-capped at 72 W.** The wide model's advantage is throughput, so it may shrink or vanish on hardware
  where the 11 x 576 matrices already run efficiently.
- **Other things change with the shape:** head count (head dimension fixed at 64), non-embedding parameters (42.5M to
  45.8M; this favours the deep model per token, which still lost), and nanoGPT's depth-scaled initialisation.

## What happened later / follow-ups
- **Phase 6 (E08):** recipe RA = Muon + 6 x 768 + 16k vocabulary; RB = the same with 11 x 576. Width given Muon and 16k:
  -0.0087 (alone -0.0177). The wide recipe kept its speed advantage, but each extra token is worth less once Muon learns
  faster per step. RA vs RB -0.0087 on validation and -0.0089 on test: just inside the declared threshold (0.0091), so RB
  is the declared final recipe by the tie-break fixed before the runs (fewer changes; no L4-specific gain). Under the
  recipes' own, smaller seed variance RA is better (every RA seed below every RB seed); the report gives both.
- **Muon's best LR at width 768 was 4e-3** (2e-3 at 576): shape and optimiser settings interact.
- **Not done:** more seeds for 6 x 768 alone; other widths, sizes or GPUs; the Muon + 6 x 768 cell without the 16k
  vocabulary.

## Compute and time used
| Item | Jobs | Counted GPU time | Wall time | Ledger cost |
|---|---|---|---|---|
| LR pilots (4 values x 2 shapes at seed 0; close calls 2 values x 2 seeds x 2 shapes) | 16 | 4.0 h | 4.4 h | $3.51 |
| 60-minute runs (seed 0 per shape at 2e-3) | 2 | 2.0 h | 2.1 h | $1.66 |
| Baseline (E02-N49, reused) | 0 | - | - | - |
| **Total** | **18** | **6.0 h** | **6.5 h** | **$5.17** |

Runs on 5 Nov 2026, 10:40 to ~17:00, on jinkyungjeon (commit 7e1b431). The ledger counts GPU time only (Modal bills ~1.2x
in total).

## How to reproduce
- Configs: `configs/E06/{W6x768,D19x448}_lr{5e-4,1e-3,2e-3,4e-3}_15min.yaml` (seed 0; W6x768 1e-3 and 2e-3, D19x448 2e-3
  and 4e-3 also seeds 1, 2); `W6x768_lr2e-3_60min.yaml` and `D19x448_lr2e-3_60min.yaml` (seed 0). The other 60-minute files
  exist because one config was written per LR before the pilots; they were not run. Baseline: `configs/E02/N49.yaml`.
- Tables and decisions: `python code/summarize.py E06`; test: `python code/summarize.py TEST`.
- Result files: `results/runs/E06-*.json`, `results/runs/E01-N49_*.json`, `results/runs/E02-N49__s*.json`,
  `results/test/E06-*.json`. Decisions D28, D30, D32-D34; DECLARATIONS Amendments row of 5 Nov.

## How this phase supports the report (marking criteria)
| Criterion | Evidence from E06 |
|---|---|
| Experimental design and exploration (35 %) | Parameter count held within 2 %; shapes bracket Kaplan's flat range (width/depth 24, 52, 128); the full LR protocol per shape (upgraded from 2 values before any run); a declared seed-0 rule with a close-call fallback; the single-seed wide result re-tested inside the Phase 6 recipe |
| Understanding and problem formulation (25 %) | Per-token learning separated from throughput: width wins through GPU efficiency, not learning; contrasted with Kaplan et al. and MobileLLM; the hardware dependence stated |
| Analysis and report quality (20 %) | Machine-speed adjustment, four sets and test; the single-seed caveat stated and followed up; missed prediction reported |
| Implementation and reproducibility (20 %) | One config per LR (nothing edited between rounds); the missing slow-machine check fixed before any run; every number traced to run and test files |

