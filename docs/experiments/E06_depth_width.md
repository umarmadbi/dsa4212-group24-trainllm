# E06: Deep-and-narrow or shallow-and-wide at the same size? (Phase 4b, shape)


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

## Findings
- At equal parameters and equal time on the L4, fewer wider layers win: 6x768 < 11x576 < 19x448 (width/depth 128 > 52 > 24).
- The wide model's advantage is throughput, not per-token learning: +16 % tokens/s (MFU 0.26 vs 0.22) and a tie at equal steps.
  Hardware-specific: it reflects how efficiently this GPU runs wide vs narrow matmuls.
- The deep model loses both per step and in speed; MobileLLM's deep-thin advantage does not appear at 49M under a time budget.
- Prediction ("no shape clearly better") missed. Limitation: the wide verdict rests on one seed (declared rule; 1.3x the threshold).

