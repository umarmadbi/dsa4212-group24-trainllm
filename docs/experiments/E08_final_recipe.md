# E08: the combined recipe and the final test evaluation (Phase 6)

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

## Budget, placement and order (credits ~12:00 6 Nov, to be confirmed: giraffes573 ~$2.9, joshuahanrc ~$2.8, jinkyungjeon ~$10.6, umarmadbi ~$2)
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

## Surprises, failures and what we changed
- RA's speed: the recipe kept width's full +16 % speed advantage (D43; predicted to shrink). Its BPB gain from width halved anyway.
- RA vs RB missed the declared threshold by 0.0004 on val and 0.0002 on test; RB stays the declared final (D45).
- First test-scoring attempt cancelled by a dropped connection (`.remote()` bug, D46); fixed with `spawn`; no model scored twice.


