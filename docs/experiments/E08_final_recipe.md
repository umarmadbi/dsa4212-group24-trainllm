# E08: the combined recipe and the final test evaluation (Phase 6)

## Summary
- **Question:** the three changes that each beat the baseline on their own (Muon, the wide 6 x 768 shape, a 16k vocabulary)
  had never been run together. How good is the best recipe built from them in the same 60 minutes on one L4, do their gains
  add up, and do all our conclusions hold on the untouched test set?
- **Answer:** **both recipes beat the baseline clearly, on validation and on test.** RB = Muon + 16k (11 x 576) **1.1531 val /
  1.1583 test, -0.041**; RA = Muon + 16k + 6 x 768 **1.1444 val / 1.1494 test, -0.050** (baseline 1.1945 / 1.1994; 3 seeds
  each; threshold 0.0091). Both also beat the best single change (Muon alone, -0.026) and are better on every validation and
  test set.
- **Final recipe:** RA vs RB is -0.0087 on validation (-0.0089 on test), just inside the declared threshold, so **RB is the
  declared final recipe** by the tie-break fixed before the runs. **RA is the best model we measured**, and it is better than
  RB beyond noise if judged by the recipes' own seed spread (2 SE 0.0034; every RA seed below every RB seed). The report gives
  both readings.
- **Do the gains add up?** Muon and the 16k vocabulary add up almost exactly (interaction +0.001). Width keeps its whole
  +16 % speed advantage inside the recipe, but its BPB gain halves (-0.009 vs -0.018 alone): extra tokens are worth less
  once Muon learns more per step. Combined, the recipes are worth 1.6x (RB) to 1.9x (RA) the best single change.
- **Held on every check:** machine-speed adjustment (RA -0.048, RB -0.038), overlapping-window scoring (~93 % of each gain
  remains), removing contaminated test documents (changes nothing by more than 0.0008), and an embedding-LR check under Muon
  (the inherited 2e-3 was the best of three).
- **Test set (scored once, for all 57 reported models):** test = validation + 0.005 for every model, the same ranking
  (Spearman 1.000). No decision anywhere in the project used a test score.
- **Prediction:** held for every BPB quantity (both recipe scores, the RA-RB near-miss, sub-additivity, val-test agreement);
  missed for RA's speed (it kept width's full speed advantage).
- **Figures:** `report/figures/recipe_attribution.pdf` (`code/plots/fig_recipe_attribution.py`): each change alone and
  combined vs the additive prediction, validation and test; `report/figures/val_vs_test.pdf` (`code/plots/fig_val_vs_test.py`);
  `report/figures/recipe_and_test.pdf` (`code/plots/fig_recipe_and_test.py`) combines both.
- **Cost:** 18 fifteen-minute pilots, 6 sixty-minute finals, one-time test scoring of 57 models; 10.5 counted GPU-hours + 1.0 h
  of scoring, ledger $9.74 (GPU only), all on 6 Nov 2026.

## Goal
Phases 2-5 changed one thing at a time against a fixed baseline. That tells us which changes help on their own, not whether
they still help together: two changes can fix the same weakness (and add less than their sum) or get in each other's way
(one change costs the time another one needs). The Smol Training Playbook warns about exactly this: a combined gain does not
prove each ingredient helps, and changing several factors at once identifies a better recipe but not which part caused it.
So Phase 6 builds the combined recipe the plan promised ("baseline vs best combined recipe, 3 seeds each, test set once"),
and is designed so the combination can also be taken apart again. It then closes the project with the one-time test
evaluation of every model the report quotes.

## Question
Phases 4 and 5 found three changes that each beat the declared baseline (AdamW, 11 layers x 576, 8k vocabulary,
1.1945 BPB) on their own: **Muon** (-0.026, 3 seeds), **6 wide layers instead of 11** (-0.018, 1 seed, all from running
16 % faster on the L4) and **a 16k or 32k vocabulary** (-0.016 / -0.017). They were never tried together.
(1) How good is the best recipe we can build from them, under the same 60-minute budget? (2) Do the gains add up, or
do they overlap or get in each other's way? (3) How do all the reported models do on the untouched test set?

## Why the gains might not add up (what this experiment has to check)
| Interaction | Why it matters |
|---|---|
| Muon's extra work per step grows with width | Newton-Schulz cost index 25.2 (11x576) vs 32.6 (6x768): +29 %. Muon slowed 11x576 by 3-5 %; on 6x768 expect ~5-6 % |
| The vocabulary's extra work grows with width | Output layer = width x vocabulary: 16k at 768 wide costs 33 % more than at 576. 16k slowed 11x576 by 10.5 % |
| The wide shape's gain IS speed | E06: equal per step, +16 % tokens. Muon and 16k both eat into that speed, so the wide gain may shrink or vanish inside the recipe |
| Muon learns faster per step | A faster learner may get less out of extra steps (diminishing returns), shrinking the wide gain further |
| Embedding learning rate | Under Muon the embedding stays on AdamW at 2e-3, tuned for AdamW, never tuned under Muon (D36 gap 4). 16k doubles the embedding (V x width) |
| Overlap of mechanisms | Muon and a larger vocabulary may fix partly the same thing (e.g. faster learning of rare-token rows), so each could add less in the presence of the other |

## Design (D42)
| Item | Choice | Why |
|---|---|---|
| Recipe RA | Muon + 6x768 + 16k vocabulary (55.5M total, 12.6M embedding, 42.5M non-embedding) | Every REAL winner together: the natural "best combined recipe" |
| Recipe RB | Muon + 11x576 + 16k (53.5M total, 9.4M embedding, 43.8M non-embedding) | Tests whether width still pays inside the recipe (its gain is speed, which Muon and 16k erode); gives a clean Muon x vocabulary interaction from runs that exist already; the safer recipe if width's gain vanishes |
| Vocabulary | **16k, not 32k** | 16k and 32k are tied (0.0007 apart, D40), so a declared tie-break: 16k has 3 seeds (32k has 1); costs less per token (-10.5 % vs -25 % tokens/s), and the gap grows with width (32k x 768 = 25M embedding params, output layer ~53 % of the block FLOPs); a smaller embedding is less exposed to the untuned embedding LR under Muon; the evaluation-window caveat (D39) is the same ~1/3 for both. 32k at the grid edge stays a reported limitation |
| Everything else | as the baseline: FineWeb-Edu, context 512, 32,768 tokens/step, warmup 200, cosine to 10 % by time, weight decay 0.1, clip 1.0, compile, bf16, 60 counted min on one L4 | One recipe = only the three changes |
| Optimiser details | Muon on the 4 hidden matrices of every block (fused qkv as one), PyTorch Muon (momentum 0.95 Nesterov, 5 NS steps, update RMS matched to AdamW); AdamW (0.9, 0.95) at 2e-3 for embeddings + LayerNorm | Exactly E05's arm |
| Data | `/data/vocab/v16384/` (built on giraffes573 and joshuahanrc, verified); jinkyungjeon (and umarmadbi if used) build + `verify` first | Same files everywhere, checked by hash in every run |
| Configs | `configs/E08/`: per recipe, muon_lr 2.5e-4 ... 3.2e-2 x {15, 60} min; RA's R1e configs (`_emb1e-3`, `_emb4e-3`) | One file per LR: nothing edited between rounds (no placeholders, D30) |

### Rules (fixed before any E08 run)
- **R0 data and infrastructure.** As E07: every run's variant hashes must equal the verified manifest; committed code,
  NVIDIA L4, one run per container, cold compile, full budget; slow machine = > 10 % below that recipe's own pilots ->
  excluded until replaced (same seed, `--fresh`); never replaced because of its result.
- **R1 learning rate (per recipe), same effort as E05.** muon_lr 1e-3, 2e-3, 4e-3, 8e-3 x 15 min, seed 0; edge rule (next
  factor-2 value beyond the edge, at most 2 per side); close call (runner-up within 2.8 sigma15 = 0.0188) -> seeds 1, 2 of
  both, lower 3-seed mean wins. The AdamW part stays at 2e-3 (as E05). No horizon check (E05 checked Muon's 15-min pick at
  60 min and it held; E06/E07 had none).
- **R1e embedding-LR check (diagnostic, never changes the recipe).** RA at its pick, AdamW part at 1e-3 and 4e-3, 15 min,
  seed 0, against RA's pick at 2e-3. If one is lower by > 2.8 sigma15 the report says the recipe is under-tuned in that knob
  and its gain is a lower bound. Why not decisive: the baseline had one LR knob tuned; giving the recipe a second tuned knob
  would give it more tuning effort than the baseline (biased towards the recipe).
- **R2 finals.** Each recipe at its R1 pick, 60 min, seeds 0, 1, 2 (all three: Phase 6 is the declared 3-vs-3 comparison).
- **R3 decisions** (FineWeb-Edu validation BPB, 3 vs 3, real if |difference| > 1.6 sigma60 = 0.0091, sigma60 = 0.0057 as declared):
  1. each recipe vs the baseline (E02-N49 s0-2, 1.1945): headline;
  2. each recipe vs Muon alone (E05, 1.1683, the best single change): does combining beat the best single change?
  3. **final recipe** = RA if RA is lower than RB beyond the threshold, RB if RB is lower beyond it, **RB if they are not
     separated** (tie-break: fewer changes from the baseline; its gain does not rest on L4-specific speed). If RB ends with
     fewer than 3 usable seeds (credits), the final recipe is RA.
- **Reported beside the decisions, never decisive:** speed (tokens/s, text/s, MFU) vs the baseline and every single change;
  per-time vs per-byte view (D3, learning curves on the same text as the 8k baseline); machine-speed sensitivity (R4b);
  AVG4 and the four sets; clipping; memory; the 15-min comparison; D8 overlapping-window rescore of all six finals (stride
  128, FineWeb-Edu val, cents); the R1e check; Welch-type view with each arm's own seed sd (Muon runs vary ~5x less than
  AdamW runs, so the declared sigma60 is conservative for them); **attribution table** (D-attribution in `summarize.py
  E08`): with B = baseline, M = Muon, W = 6x768, V = 16k: M-B, W-B, V-B (existing), MV-M (16k given Muon), MV-V (Muon given
  16k), MWV-MV (width given Muon + 16k), MV-M-V+B (Muon x 16k interaction; 0 = additive), MWV-M-W-V+2B (non-additivity of
  all three), each with 2 SE from sigma60 and the seed counts (W has 1 seed, so terms with W are wide). Cells not run
  (M+W, W+V): budget; stated as a limitation.

### Test phase (after every R3 decision; the test lock)
- `configs/frozen_configs.yaml` (drafted now, status `draft`): every model the report quotes. Tiers: **core** (baseline,
  E05 Muon, E06 shapes, E07 vocabularies, E08 finals: four test sets + without contaminated documents + overlapping
  windows on FineWeb-Edu); **data** (E03 arms: four sets + without contaminated documents; AVG4 is that comparison's
  declared metric); **extra** (E02 sizes, E04 repetition: FineWeb-Edu only; lowest priority).
- **Freeze:** after R3, add the E08 finals of both recipes (s0 with text samples), set `status: frozen`, commit; only then
  `modal run --detach modal_app.py::test_eval` once per workspace (`--tiers core,data` first if credits are thin).
- The lock in code: `evaluate_test.py` refuses unlisted runs and a non-frozen list; `test_eval` refuses uncommitted code or
  frozen list and any run already scored (laptop or Volume); the container re-checks that its frozen list equals the
  laptop's. Each test result records the frozen list's hash and commit.
- **Contaminated test documents** (> 1 % of the FineWeb-Edu, FineWeb and DCLM test sets, declared 13-gram rule): every
  test score is also given without them (the same documents for every model), as rule R3 did for validation.
- **What is reported:** the final recipe vs the baseline on FineWeb-Edu test (3 vs 3, mean +- sd, difference with the
  declared noise scale), the same without contaminated documents and with overlapping windows; every frozen model's val
  and test side by side; val-test rank agreement (Spearman); E03 arms on AVG4 test. **No decision is made or revised from
  test scores**; a val/test disagreement is reported as such.
- **Text samples** (illustration only, D42): 3 fixed prompts, temperature 0.8, top-k 50, 100 new tokens, seed 1234 + prompt
  index; baseline s0 and both recipes' s0.

## Gaps checked before any run (D41/D42) and how each is handled
| Gap | Handling |
|---|---|
| Three winners never run together; unknown interactions | RA and RB, 3 seeds each, plus the attribution table using the existing single-change runs |
| The wide shape's REAL verdict rests on one seed at 1.3x the threshold | RA vs RB measures width with 3 vs 3 inside the recipe (replaces the extra E06 seeds promised in D34) |
| Muon's LR was tuned only at 11x576/8k | Full R1 protocol per recipe |
| Embedding LR never tuned under Muon; 16k doubles the embedding | R1e diagnostic (2 pilots) |
| 16k vs 32k is a tie | Declared tie-break with reasons (above); 32k reported as a limitation |
| ~1/3 of a vocabulary gain is the token-based scoring window (D39) | D8 rescore of the E08 finals on val, and overlapping windows on test for every core model; the report gives both |
| Choosing the best recipe on val makes its val score optimistic (winner's curse) | The test score is the unbiased one and is reported for both recipes |
| Old references (baseline 4 Nov, Muon 5 Nov, 16k 6 Nov) on other workspaces and commits | Training code (train.py, model.py, bpb.py, config.py) unchanged since afb895c (D36); speed checks vs own pilots; R4b machine-speed sensitivity; data hashes |
| Many comparisons in the project, each at ~2 SE | The final 3-vs-3 comparison is the confirmation; stated in the report |
| Test-set contamination > 1 % on three sets | Contaminated documents removed in a second score, same documents for every model |
| Test set touched before the last decision | Lock in code + git history; scoring only after the freeze commit |
| Credits are per workspace and small | Allocation and fallbacks below; the 4k close-call seeds dropped (D42) |
| Ledger cost = GPU time only | Report costs from Modal usage pages (D36) |

## Prediction (D42, before any run)
- R1: both recipes pick muon_lr 2e-3 (or a close call 2e-3 vs 4e-3, as E05).
- Speed (tokens/s vs baseline): RA -2 to -8 %; RB -12 to -17 % (Muon -4 % x 16k -10.5 %).
- 60-min val BPB: RB 1.148-1.158 (additive estimate 1.152); RA 1.138-1.152 (additive 1.134). Gains sub-additive by
  0.003-0.010. RA - RB between -0.004 and -0.012: most likely inconclusive (then RB is the final recipe by the tie-break).
- Both recipes BETTER than the baseline (by ~0.04-0.05) and than Muon alone (by ~0.012-0.025).
- R1e: 2e-3 within noise.
- Test: the same ordering as val; test-val offsets within ~0.005 of each other across models; Spearman rho > 0.9.

## Budget, placement and order (the plan, written before the runs; what was actually used is in "Compute and time used"; credits ~12:00 6 Nov: giraffes573 ~$2.9, joshuahanrc ~$2.8, jinkyungjeon ~$10.6, umarmadbi ~$2)
Real cost ~$1.2 per 60-min run, ~$0.3 per 15-min pilot, test scoring ~$0.02-0.05 per model (one container per workspace).
| Round | What | Where | Cost |
|---|---|---|---|
| 0 (now, parallel) | v16384 build + verify | jinkyungjeon (needed), umarmadbi (optional: only for a second close call) | ~$0.15 each |
| 0 | N26 s0 val rescore on FineWeb, DCLM, Wikipedia (long-open item) | umarmadbi | cents |
| 0 | D8 rescore of E07 16k s1, s2 (completes the 3-seed sliding view) | joshuahanrc | cents |
| 1 | RA pilots (4) | giraffes573 | ~$1.2 |
| 1 | RB pilots (4) | joshuahanrc | ~$1.2 |
| 2 (only if needed) | close-call pilots: first on jinkyungjeon; a second recipe's on umarmadbi | | ~$1.2 each |
| 3 | RA s0-2 and RB s0-2 (60 min) | jinkyungjeon | ~$7.2 |
| 3 | R1e (2 pilots) | giraffes573 | ~$0.6 |
| 4 | D8 rescore of the six finals; R3 decisions; freeze + commit | jinkyungjeon | cents |
| 5 | test_eval per workspace (core, data, then extra) | all four | ~$1-1.5 |
Expected total ~$13-14 of ~$18; jinkyungjeon ends ~$1.5 above zero (worst case ~$0.3: then one RB final moves to
joshuahanrc and joshuahanrc's `extra` tier is skipped). Priority if credits fall short: (1) R2 finals of RA, (2) R2 finals
of RB, (3) test scoring core + data, (4) R1e, (5) extra tier. **Dropped: E07 4k seeds 1, 2** (can only end "inconclusive" or
"worse"; reported as a single-seed close call, D42). Time (planned on 6 Nov): R1 ~1.5-2 h incl. queue, R2 ~1.5-2 h, R3 ~2-3 h: E08 decided
that night or the next morning, test scoring 7 Nov, well before the ~10 Nov end of experiments (in the event, all of it finished on 6 Nov).

## Runs
| Run ID | Config | Seed | Commit | Counted time (s) | Tokens | Final val BPB | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| E08-W6x768_V16384_muon_lr1e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.2627 | done | giraffes573; 89.1k tok/s; clipped 5.4 % (start only) |
| E08-W6x768_V16384_muon_lr2e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.2645 | done | giraffes573; 82.5k tok/s |
| E08-W6x768_V16384_muon_lr4e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.2625 | done | giraffes573; 84.4k tok/s; best s0 |
| E08-W6x768_V16384_muon_lr8e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.3896 | done | giraffes573; progress stalls steps ~500-1,100 |
| E08-N49_V16384_muon_lr1e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.2923 | done | joshuahanrc; clipped 10.2 % (start only) |
| E08-N49_V16384_muon_lr2e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.2711 | done | joshuahanrc; best s0 |
| E08-N49_V16384_muon_lr4e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.2813 | done | joshuahanrc |
| E08-N49_V16384_muon_lr8e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.3073 | done | joshuahanrc |
| E08-W6x768_V16384_muon_lr1e-3_15min | configs/E08/ | 1 | 776406f | 900 | 15 min | 1.2619 | done | jinkyungjeon; 88.6k tok/s; close call (D44) |
| E08-W6x768_V16384_muon_lr1e-3_15min | configs/E08/ | 2 | 776406f | 900 | 15 min | 1.2668 | done | jinkyungjeon; 84.9k tok/s; close call (D44) |
| E08-W6x768_V16384_muon_lr4e-3_15min | configs/E08/ | 1 | 776406f | 900 | 15 min | 1.2661 | done | jinkyungjeon; 81.6k tok/s; close call (D44) |
| E08-W6x768_V16384_muon_lr4e-3_15min | configs/E08/ | 2 | 776406f | 900 | 15 min | 1.2597 | done | jinkyungjeon; 85.7k tok/s; close call (D44) |
| E08-N49_V16384_muon_lr2e-3_15min | configs/E08/ | 1 | 776406f | 900 | 15 min | 1.2735 | done | jinkyungjeon; 76.2k tok/s; close call (D44) |
| E08-N49_V16384_muon_lr2e-3_15min | configs/E08/ | 2 | 776406f | 900 | 15 min | 1.2759 | done | jinkyungjeon; 74.4k tok/s; close call (D44) |
| E08-N49_V16384_muon_lr4e-3_15min | configs/E08/ | 1 | 776406f | 900 | 15 min | 1.2865 | done | jinkyungjeon; 71.3k tok/s; close call (D44) |
| E08-N49_V16384_muon_lr4e-3_15min | configs/E08/ | 2 | 776406f | 900 | 15 min | 1.2845 | done | jinkyungjeon; 72.1k tok/s; close call (D44) |
| E08-W6x768_V16384_muon_lr4e-3_emb1e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.2702 | done | giraffes573; R1e |
| E08-W6x768_V16384_muon_lr4e-3_emb4e-3_15min | configs/E08/ | 0 | 776406f | 900 | 15 min | 1.2659 | done | giraffes573; R1e |
| E08-W6x768_V16384_muon_lr4e-3_60min | configs/E08/ | 0 | 776406f | 3600 | 293M | 1.1453 | done | jinkyungjeon; RA final |
| E08-W6x768_V16384_muon_lr4e-3_60min | configs/E08/ | 1 | 776406f | 3600 | 308M | 1.1428 | done | jinkyungjeon; RA final |
| E08-W6x768_V16384_muon_lr4e-3_60min | configs/E08/ | 2 | 776406f | 3600 | 297M | 1.1449 | done | jinkyungjeon; RA final |
| E08-N49_V16384_muon_lr2e-3_60min | configs/E08/ | 0 | 776406f | 3600 | 253M | 1.1537 | done | jinkyungjeon; RB final |
| E08-N49_V16384_muon_lr2e-3_60min | configs/E08/ | 1 | 776406f | 3600 | 252M | 1.1554 | done | jinkyungjeon; RB final |
| E08-N49_V16384_muon_lr2e-3_60min | configs/E08/ | 2 | 776406f | 3600 | 267M | 1.1502 | done | umarmadbi; RB final |

## Results
| Arm | Seeds | FWE val BPB (mean) | vs baseline | tokens/s vs baseline | text read |
|---|---|---|---|---|---|
| Baseline (AdamW 11x576 8k) | 3 | 1.1945 | - | - | 1.127 GB |
| Muon alone (E05) | 3 | 1.1683 | -0.026 | -3 % | |
| 16k alone (E07) | 3 | 1.1781 | -0.016 | -10.5 % | |
| 6x768 alone (E06) | 1 | 1.1768 | -0.018 | +16 % | |
| **RB** Muon + 11x576 + 16k (declared final) | 3 | **1.1531** | **-0.041** | -14 % | 1.099 GB |
| **RA** Muon + 6x768 + 16k (best measured) | 3 | **1.1444** | **-0.050** | +1 % | 1.279 GB |

### Learning-rate pilots (15 min; seed 0, 3-seed mean in brackets where a close call was run)
| muon_lr | 1e-3 | 2e-3 | 4e-3 | 8e-3 | Pick |
|---|---|---|---|---|---|
| RA (6 x 768) | 1.2627 (1.2638) | 1.2645 | **1.2625 (1.2627)** | 1.3896 (stalls) | 4e-3, close call vs 1e-3 (gap 0.0011): flat basin 1e-3-4e-3 |
| RB (11 x 576) | 1.2923 | **1.2711 (1.2735)** | 1.2813 (1.2841) | 1.3073 | 2e-3, close call vs 4e-3 (gap 0.0106), as E05 |

R1e (RA at muon_lr 4e-3, AdamW part for embeddings + LayerNorm): 1e-3 1.2702, **2e-3 1.2627 (3 seeds)**, 4e-3 1.2659: the
inherited 2e-3 is the best tried, so the recipe is not under-tuned in that knob.

### Speed and text read (pilots: median steady tokens/s; finals: 3-seed means)
| Arm | tokens/s vs baseline | text/s vs baseline | tokens in 60 min | text read | MFU |
|---|---|---|---|---|---|
| Baseline (AdamW 11 x 576 8k) | - | - | 291M | 1.127 GB | 0.222 |
| RB (Muon + 11 x 576 + 16k) | -13.7 % | -4.8 % | 257M | 1.099 GB | 0.214 |
| RA (Muon + 6 x 768 + 16k) | +1.0 % | +11.3 % | 299M | 1.279 GB | 0.248 |
RA vs RB +16 % tokens/s, the same as 6 x 768 vs 11 x 576 under AdamW: Muon + 16k cost both shapes ~14 %.

### Where each recipe's gain comes from (D3, learning curves on the same text as the 8k baseline)
| | at equal time (60 min) | at equal text read | worth of reading more/less text per hour |
|---|---|---|---|
| RA - baseline | -0.050 | -0.038 | -0.013 |
| RB - baseline | -0.041 | -0.043 | +0.002 |
RB's gain is entirely "more learned per byte" (Muon per step, 16k per byte); RA adds ~0.013 from reading 14 % more text.

### Attribution: each change alone and inside the recipes (FineWeb-Edu val; B baseline, M Muon, W 6 x 768, V 16k)
| Quantity | Value | 2 SE | Reading |
|---|---|---|---|
| M - B (Muon alone) | -0.0262 | 0.0093 | |
| W - B (6 x 768 alone, 1 seed) | -0.0176 | 0.0132 | |
| V - B (16k alone) | -0.0163 | 0.0093 | |
| MV - M (16k given Muon) | -0.0152 | 0.0093 | ~ alone |
| MV - V (Muon given 16k) | -0.0251 | 0.0093 | ~ alone |
| MWV - MV (6 x 768 given Muon + 16k) | -0.0087 | 0.0093 | half of alone |
| MV - M - V + B (Muon x 16k interaction) | +0.0011 | 0.0132 | additive |
| MWV - M - W - V + 2B (all three) | +0.0100 | 0.0208 | mildly sub-additive (all from width) |
Not run (budget): M+W and W+V, so the width x Muon and width x vocabulary interactions cannot be separated.

### Robustness of the headline (3 vs 3, difference to the baseline)
| Check | RA | RB | RA - RB |
|---|---|---|---|
| Validation, declared | -0.0501 | -0.0414 | -0.0087 |
| Machine-speed adjusted (R4b) | -0.0475 | -0.0376 | |
| Overlapping windows, val (D8) | -0.0466 | -0.0386 | -0.0080 |
| AVG4 val (four sets) | -0.0510 | -0.0406 | |
| Per val set FWE / FineWeb / DCLM / Wikipedia | -0.050 / -0.046 / -0.051 / -0.057 | -0.041 / -0.038 / -0.040 / -0.044 | |
| Test, declared | -0.0500 | -0.0411 | -0.0089 |
| Test without contaminated documents | -0.0492 | -0.0404 | -0.0088 |
| Test, overlapping windows | -0.0469 | -0.0386 | -0.0082 |
| Per test set FWE / FineWeb / DCLM / Wikipedia | -0.050 / -0.048 / -0.051 / -0.055 | -0.041 / -0.038 / -0.040 / -0.042 | |
| Seed sd (val / test) | 0.0013 / 0.0011 | 0.0027 / 0.0025 | baseline 0.0014 / 0.0015 |
| 15 min, 3 seeds (descriptive) | -0.088 | -0.077 | -0.011 |

### The RA vs RB learning curves (3-seed means, same text subset)
RA is ahead early (-0.019 at 10 % of the hour), level from 30 % to 70 %, then pulls ahead during the final LR decay (-0.004 /
-0.006 / -0.009 at 80 / 90 / 100 %). No late stall at RA's higher LR (4e-3), the risk noted in D44.

## Findings
- Both recipes beat the baseline (-0.050 / -0.041) and Muon alone, on every validation set and after the machine-speed adjustment.
- RA vs RB -0.0087: just inside the declared threshold (0.0091) -> RB final by tie-break; beyond noise under the recipes' own seed
  variance (2 SE 0.0034). Both reported, both tested.
- Muon and the 16k vocabulary add up almost exactly (interaction +0.001). Width keeps its speed advantage in the recipe but its
  gain halves (-0.009 vs -0.018): extra tokens are worth less to a faster-learning optimiser.
- The embedding LR inherited from AdamW (2e-3) is the best of 1e-3 / 2e-3 / 4e-3 under Muon (R1e).

### Test set (scored once, D47)
| Arm | Test FWE BPB | vs baseline | w/o contaminated docs | overlapping windows |
|---|---|---|---|---|
| Baseline | 1.1994 +- 0.0015 | - | - | - |
| RB (declared final) | 1.1583 +- 0.0025 | -0.0411 | -0.0404 | -0.0386 |
| RA (best measured) | 1.1494 +- 0.0011 | -0.0500 | -0.0492 | -0.0469 |
Test = val + 0.005 for every model (57); Spearman 1.000; RA - RB -0.0089 on test (as on val).

### Text samples (illustration only; fixed prompts, temperature 0.8, top-k 50, 100 tokens; `python code/summarize.py TEST`)
All three models (baseline, RA, RB, seed 0) write fluent, on-topic English with frequent factual nonsense and repetition (e.g.
the baseline repeating "Subtracting (S)" for the quadratic-equation prompt). A 0.04-0.05 BPB difference is not visible in three
samples; they show what a ~50M model trained for one hour can do, not which recipe is better.

### Outcome of every declared rule
| Rule | Fired? | Outcome |
|---|---|---|
| R0 data and infrastructure | Clean | All 24 E08 training jobs: committed code, NVIDIA L4, one attempt, cold compile, full budget, data hashes = verified manifests; speed vs own pilots -3.2 % to +3.5 %; none replaced |
| R1 LR, RA | Yes | Close call 4e-3 vs 1e-3 -> seeds 1, 2 -> **4e-3** (1.2627 vs 1.2638); no edge |
| R1 LR, RB | Yes | Close call 2e-3 vs 4e-3 -> seeds 1, 2 -> **2e-3** (1.2735 vs 1.2841); no edge |
| R1e embedding LR (diagnostic) | Yes | 2e-3 best (1e-3 +0.0075, 4e-3 +0.0032): not under-tuned |
| R2 finals | Yes | 3 usable seeds per recipe |
| R3.1 vs baseline | Yes | RA -0.0501, RB -0.0414 -> both **BETTER** |
| R3.2 vs Muon alone | Yes | RA -0.0239, RB -0.0152 -> both **BETTER** |
| R3.3 final recipe | Yes | RA - RB -0.0087 < 0.0091 -> not separated -> **RB** (tie-break) |
| Fallbacks (RB < 3 seeds; credit priority list) | No | Not needed; every planned run fitted |
| Test lock | Yes | Frozen list committed at 3ba1d90 (57 runs) before any test score; each run scored once (D46: a cancelled first attempt re-scored nothing) |

### Prediction check (D42, written before any E08 run)
| Prediction | Outcome | |
|---|---|---|
| Both pick muon_lr 2e-3 (or close call 2e-3/4e-3) | RB 2e-3 (close call 2e-3/4e-3); RA 4e-3 (close call 4e-3/1e-3) | half held |
| Tokens/s: RA -2 to -8 %, RB -12 to -17 % | RA +1.0 %, RB -13.7 % | RA missed, RB held |
| RB 1.148-1.158; RA 1.138-1.152 | 1.1531; 1.1444 | held |
| Sub-additive by 0.003-0.010 | +0.0100 (all three); Muon x 16k +0.0011 | held |
| RA - RB -0.004..-0.012, most likely inconclusive -> RB final | -0.0087, inconclusive -> RB | held |
| Both BETTER than baseline (~0.04-0.05) and than Muon alone | -0.050 / -0.041; -0.024 / -0.015 | held |
| R1e within noise | 2e-3 best | held |
| Test ordering = val, offsets within ~0.005, Spearman > 0.9 | Offsets +0.0046-0.0062, Spearman 1.000 | held |

## Surprises, failures and what we changed
- RA's speed: the recipe kept width's full +16 % speed advantage (D43; predicted to shrink). Its BPB gain from width halved anyway.
- RA vs RB missed the declared threshold by 0.0004 on val and 0.0002 on test; RB stays the declared final (D45).
- First test-scoring attempt cancelled by a dropped connection (`.remote()` bug, D46); fixed with `spawn`; no model scored twice.
- **RA's best Muon LR moved to 4e-3** (RB and E05 at 11 x 576: 2e-3). The basin is flat (1e-3 to 4e-3 within 0.002), so
  this is weak evidence that the LR shifts with width, but it shows why re-tuning per recipe was worth its cost (D44).
- **RA at muon_lr 8e-3 stalled** (validation progress flat from step ~500 to ~1,100; no spike or divergence). Muon at
  11 x 576 lost only 0.025 at 8e-3: the wide shape tolerates high LRs less (D43).
- **Clipping of 5-10 % of steps at muon_lr 1e-3** (both recipes), all in the first ~165 steps, none afterwards: the start-up
  gradient norm (~9) takes longer to fall below the clip at a low LR. Not an instability (as E05).
- **The "1/3 of a vocabulary gain is the scoring window" statement was corrected** with 3 seeds of 16k (D43): the window effect
  is a near-constant ~0.003 BPB, ~21 % of 16k's gain; the 1/3 came from seed 0, whose gap was the smallest.
- **Infrastructure:** the test-scoring entry point first used `.remote()`, which Modal cancels when the laptop's connection
  drops, even with `--detach`; launch and rescore had used `spawn` for this reason since D16. Fixed in D46 (spawn, per-model
  errors caught). Results already saved on the Volume were kept and refused a second scoring, so the lock held. umarmadbi's
  six models were scored from the commit before the fix (that attempt was not cut off; scoring code identical).
- **Dropped on purpose:** E07's 4k close-call seeds (can only end "inconclusive" or "worse"; the credits went to RB's 3 seeds).

## How this relates to the readings
- **Smol Training Playbook (Allal et al. 2025):** change one factor at a time against a reliable baseline, then verify the
  promising choices at the final allowance, and beware that a sequentially built recipe can hide interactions. Phases 2-5
  followed the first part; E08 is the second part, and the RB arm plus the single-change runs make the combination
  decomposable (a partial factorial) instead of a single "better recipe" number.
- **Muon (Jordan et al. 2024; Liu et al. 2025):** Muon's gain survived combination almost unchanged (-0.025 given 16k vs -0.026
  alone). Liu et al.'s RMS-matched scaling lets AdamW's LR carry over: it did at 11 x 576 (2e-3 in E05 and RB); at 6 x 768 the
  pick moved to 4e-3 inside a flat basin. The topic page's question whether Muon's gain changes with architecture: its LR
  optimum may shift with width, its gain did not shrink when the vocabulary changed.
- **Kaplan et al. (2020):** at fixed size, shape matters little per step, which E06 found (equal steps: tie). Under a time
  budget the shape still matters through hardware speed; inside the recipe that speed advantage stayed but bought half as much.
- **Tao et al. (2024), vocabulary scaling:** their compute-optimal fit put our vocabulary near 5-8k; 16k beat 8k alone and
  inside the recipe. A time budget on our GPU, tied embeddings and the per-byte metric favour a larger vocabulary than their
  FLOP-based fit.
- **Geiping and Goldstein (2022), Cramming:** under a one-GPU, one-day budget most modifications they tried did not improve
  the result; what helped was what let the model learn more for the same compute. Our three changes are of that kind (more per step: Muon; more per byte: 16k; more
  steps per hour: width), and only the throughput-type change lost part of its value in combination.
- **Chinchilla (Hoffmann et al. 2022):** the recipes change neither the budget nor the size class (53-56M, inside E02's flat
  optimum of 36-65M); they show that at a fixed, already well-chosen size, the optimiser, tokenizer and shape together are
  worth -0.05, more than twice what moving from 25M (N26) into the optimal size range was worth in Phase 2 (-0.018).

## What Phase 6 says about the project question
Given 60 minutes on one L4, where should the time go? Taking all phases together (FineWeb-Edu val, differences vs the N49
baseline unless stated):
1. **Pick a size inside the optimum first** (E02): 36-65M are tied; too small costs a lot (N7 +0.088, N26 +0.018).
2. **Spend the steps better:** Muon (-0.026), the largest single gain, almost free in time.
3. **Spend the tokens better:** a 16k vocabulary (-0.016), gained per byte despite being slower per token.
4. **Spend the GPU better:** wide, shallow layers run 16 % faster on the L4 (-0.018 alone, -0.009 inside the recipe).
5. **Never repeat data beyond ~2 passes** (E04) and choose the corpus for the target text (E03: the FineWeb-Edu/DCLM mix
   is best on the four-set average but costs 0.022 on FineWeb-Edu, our declared objective, so it is not in the recipe).
The improvements in 2-4 combine to -0.041 (RB) / -0.050 (RA), 1.6-1.9x the best single one, and they hold on test.

## Limitations and what remains uncertain
- **Partial factorial:** M+W and W+V were not run, and width alone has 1 seed, so the width terms are wide and the
  width x Muon and width x vocabulary interactions cannot be separated.
- **The RA vs RB verdict depends on the noise scale:** the declared sigma60 comes from AdamW runs, which vary 2-4x more than
  the recipes. Under it RA vs RB is inconclusive (0.96 of the threshold); under the recipes' own spread RA is better.
- **One size class, one batch size, one 60-minute budget, one GPU type.** Width's advantage is L4-specific; on a GPU where
  11 x 576 already runs efficiently, RA's edge over RB would shrink.
- **16k, not 32k, was combined** (declared tie-break); 32k was at the edge of the vocabulary grid and was not combined.
- **The data-mixture winner of E03 was not combined**, by design: it is better on the four-set average but worse on
  FineWeb-Edu, the declared objective.
- **No horizon check for the recipes:** their LRs come from 15-minute pilots (Muon's 15-minute pick held at 60 minutes in E05;
  RA's learning curve shows no late stall).
- **Muon's own settings** (momentum, Newton-Schulz steps, fused qkv) were not tuned; the embedding LR was checked at one width only.
- **Quality is measured by BPB only:** no downstream task; three text samples are an illustration.

## Compute and time used
| Item | Jobs | Counted GPU time | Wall time | Ledger cost (GPU only) |
|---|---|---|---|---|
| LR pilots (4 per recipe at seed 0; close calls 2 x 2 per recipe) | 16 | 4.0 h | 4.4 h | |
| R1e embedding-LR pilots | 2 | 0.5 h | 0.5 h | $3.95 (all 18 pilots) |
| 60-minute finals (RA, RB x seeds 0-2) | 6 | 6.0 h | 6.2 h | $4.99 |
| One-time test scoring (57 models, 4 workspaces) | 4 | - | 1.0 h (scoring only) | $0.80 |
| v16384 data builds (jinkyungjeon, umarmadbi), D8 and N26 rescores | - | CPU / minutes | - | cents (not in the ledger) |
| **Total** | **28** | **10.5 h** | **12.1 h** | **$9.74** |

All on 6 Nov 2026, 12:29 to ~18:50 SGT, commits 776406f (runs), 3ba1d90 (frozen list), 7102de7 (scoring fix); workspaces
giraffes573, joshuahanrc, jinkyungjeon, umarmadbi. The ledger counts GPU time only; Modal bills ~1.2x in total (use the usage
pages for the report). The baseline and single-change arms reuse E02, E05, E06 and E07.

## How to reproduce
- Data: the v16384 variant on each workspace (`modal run --detach tools/vocab_app.py::build --vocab-sizes 16384 --no-check`, then
  `modal run tools/vocab_app.py::verify --vocab-sizes 16384`, which must print PASSED).
- Configs: `configs/E08/W6x768_V16384_muon_lr{1e-3,2e-3,4e-3,8e-3}_15min.yaml` and `N49_V16384_muon_lr{...}_15min.yaml` (seed 0;
  the close-call pairs also seeds 1, 2); `W6x768_V16384_muon_lr4e-3_emb{1e-3,4e-3}_15min.yaml` (R1e); finals
  `W6x768_V16384_muon_lr4e-3_60min.yaml` and `N49_V16384_muon_lr2e-3_60min.yaml` (seeds 0-2). The other files exist because one
  config was written per LR before the pilots; they were not run.
- Test lock: `configs/frozen_configs.yaml` (frozen at 3ba1d90), `code/evaluate_test.py`, `modal run --detach modal_app.py::test_eval`
  per workspace, then `--fetch-only`.
- Tables and decisions: `python code/summarize.py E08` (picks, decisions, attribution) and `python code/summarize.py TEST`; figures:
  `python code/plots/fig_recipe_attribution.py`, `fig_val_vs_test.py`, `fig_recipe_and_test.py`.
- Result files: `results/runs/E08-*.json`, `results/rescore/E08-*__slide128.json`, `results/test/*.json` (57). Decisions D41-D47;
  DECLARATIONS Amendments row of 6 Nov.

## How this phase supports the report (marking criteria)
| Criterion | Evidence from E08 |
|---|---|
| Experimental design and exploration (35 %) | The combination was tested rather than assumed: a second recipe (RB) turns "a better recipe" into a decomposable partial factorial; interactions predicted before the runs (speed costs at width, overlapping mechanisms); same LR protocol per recipe as every earlier configuration; every rule, tie-break and fallback fixed before the first run and applied as declared, including when the result (RA vs RB) fell 0.0004 short; the untuned embedding LR checked; a test lock built in code |
| Understanding and problem formulation (25 %) | Explains why gains add (Muon x 16k: independent mechanisms, per step vs per byte) or do not (width: a throughput gain is worth less to a faster learner); separates time, text and evaluation-window effects; answers the project question across phases; links to the Playbook, Muon, Kaplan, Tao, Cramming and Chinchilla readings |
| Analysis and report quality (20 %) | 3 vs 3 seeds with the declared threshold and the arms' own spread; machine-speed, four-set, overlapping-window and contamination checks; the near-miss reported both ways without changing the rule; test scored once for 57 models, with val-test agreement shown; prediction outcomes and limitations stated |
| Implementation and reproducibility (20 %) | No training-code change (train.py, model.py, bpb.py, config.py unchanged since D36); new test-lock code with unit tests (22/22); one config per LR; frozen list, test results and ledger rows all traceable to commits; the D46 bug and its fix recorded |
