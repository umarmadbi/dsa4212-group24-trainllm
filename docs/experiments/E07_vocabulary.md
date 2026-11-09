# E07: vocabulary size under the 60-minute budget (Phase 5)

## Question
Our tokenizer splits text into one of 8,192 pieces. A bigger vocabulary means each piece covers more text (fewer tokens per byte),
so in 60 minutes the model can read more text per token it processes and sees more text inside its 512-token window, but every token
costs more (the output layer scores every vocabulary entry) and rare pieces get little training. A smaller vocabulary is the reverse.
Which vocabulary gives the lowest bits per byte for our 49M model in 60 minutes on one L4? Was 8,192 a good choice?

## Design
| Item | Choice | Why |
|---|---|---|
| Model | N49: 11 layers x 576, AdamW, context 512, 32,768 tokens/step (the declared baseline) | Only the vocabulary changes; equal LR tuning (D36: under Muon the embedding LR was never tuned) |
| Vocabularies | 4,096, 16,384, 32,768 vs the existing 8,192 | Bracket 8k by x2 on both sides and x4 above; 16k is the one most likely to beat 8k (D36) |
| Data | `/data/vocab/v<V>/` (D31/D35): same documents and splits; same tokenizer recipe and training sample | Only the merge count differs (merges are nested) |
| Fixed | non-embedding parameters 43.8M | Kaplan/Tao convention |
| Changes with V | tied embedding 2.4M / 4.7M / 9.4M / 18.9M; total 46.5M / 48.8M / 53.5M / 63.0M; FLOPs/token 316M / 330M / 358M / 415M | Part of what "vocabulary size" means; reported |
| LR | 5e-4, 1e-3, 2e-3, 4e-3 x 15 min, seed 0 per vocabulary; edge and close-call rules | Same effort as 8k had (E01 N49) |
| Decision | 60-min seed 0 vs E02-N49 mean 1.1945: real if > 0.0132; close call -> seeds 1, 2, 3-seed rule 0.0091 | As E06 |
| Metric | FineWeb-Edu validation BPB (bits per byte is comparable across tokenizers; per-token loss is not) | Declared objective |

## What could go wrong / gaps handled (D36)
| Gap | Handling |
|---|---|
| Bigger vocabulary = more parameters | Non-embedding fixed; all three counts reported. E02 found 36-65M equally good, so the total-size change is small in effect |
| Context is 512 tokens = different amounts of text (4k ~1.75 kB, 32k ~2.34 kB) | Part of the vocabulary effect in training; the evaluation part is measured by D8 (overlapping windows, all models) |
| Compute per token grows with V | Decided at equal time (the declared budget); D2 tokens/s and bytes/s, D3 equal-bytes view |
| Learning curves cut in tokens would cover different text | Variant curves cut to the same text as 8k's (byte-matched, < 0.2 % difference, logged per run) |
| Unequal tuning | Same grid and rules as 8k; LR sensitivity printed per vocabulary |
| Data differs between workspaces | Each run hashes its files; summarize checks them against the verified manifests; mismatch -> excluded |
| Machine speed (84 % of seed noise in E02) | Slow-machine rule vs own pilots; R4b sensitivity |
| Credits | Priority order for close calls; overflow workspace; Phase 6 reserve untouched |
| 8k reference is older (4 Nov, jinkyungjeon, commit 0b7dde4) | Training code bit-identical since; same as E05/E06 used |

## Prediction (D36, before any run)
tokens/s vs 8k: 4k +2..+5 %, 16k -5..-9 %, 32k -12..-20 %. BPB vs 8k at 60 min: 4k +0.01..+0.03, 16k -0.015..+0.005, 32k -0.01..+0.02.
Best 8k or 16k. Overlapping windows help 4k most.

## Runs
| Round | Workspace | Runs | Launched | Result |
|---|---|---|---|---|
| 1 | giraffes573 | 12 pilots (V4096 / V16384 / V32768 x 4 LRs, seed 0) | 5 Nov 21:06 (afb895c); starts 22:13-23:28 | all pick 2e-3, clear; 15-min vs 8k +0.004 / -0.008 / -0.001 (D37) |
| 2 | giraffes573 | 60-min seed 0 at 2e-3 for 4k, 16k, 32k | ~00:45 (afb895c); starts 01:36-03:55 | 4k 1.2032 (+0.0087 CC), 16k 1.1841 (-0.0104 CC), 32k 1.1774 (-0.0171 BETTER) (D38) |
| 3 | joshuahanrc | 16k 60-min seeds 1, 2; D8 rescore (giraffes573, jinkyungjeon) | ~10:23 (afb895c); D8 ~10:20 | 16k 1.1734 / 1.1769 -> 3-seed 1.1781 BETTER; D8: ~1/3 of each gap is the scoring window (D39, D40) |

## Results
All runs: N49 shape (11 x 576), AdamW, peak LR 2e-3 (picked by the pilots for every vocabulary), FineWeb-Edu validation BPB of the final
model. 8k = the declared baseline E02-N49 (3 seeds, 1.1945). Source: `python code/summarize.py E07`, `results/runs/E07-*.json`,
`results/rescore/*__slide128.json`.

**Round 1 (15-min LR pilots, seed 0; D37):** 2e-3 best for all three, each clear of its runner-up by more than 0.0188 (4k 1e-3 +0.022,
16k 4e-3 / 1e-3 +0.023, 32k 1e-3 +0.038), so no edge or close-call pilots were needed. At 15 min vs the 8k 3-seed mean (1.3508,
descriptive): 4k 1.3548 (+0.004), 16k 1.3426 (-0.008), 32k 1.3498 (-0.001). One pilot (32k, 5e-4) was preempted and resumed from its
checkpoint with its counted time kept (declared resume path); it is the worst LR of its grid.

**Rounds 2-3 (60 min at 2e-3; D38, D40):**
| Vocabulary | Seeds | FWE val BPB | vs 8k (1.1945) | Verdict (declared rule) | Tokens/s vs 8k | Text read in 60 min | Total params |
|---|---|---|---|---|---|---|---|
| 4,096 | 1 | 1.2032 | +0.0087 | close call; seeds 1-2 not run (D42) | +2.5 % | -4.6 % | 46.5M |
| 8,192 (baseline) | 3 | 1.1945 | - | - | - | 1.12 GB | 48.8M |
| **16,384** | 3 | **1.1781** (1.1841 / 1.1734 / 1.1769) | **-0.0163** | **BETTER** (threshold 0.0091) | -10.5 % | -1.4 % | 53.5M |
| **32,768** | 1 | **1.1774** | **-0.0171** | **BETTER** (threshold 0.0132) | -25.0 % | -11 % | 63.0M |

- BPB falls monotonically with vocabulary size: 1.2032 / 1.1945 / 1.1841 (seed 0) / 1.1774 for 4k / 8k / 16k / 32k.
- **Robustness (reported, not decisive):** 16k speed-adjusted -0.0116 and AVG4 -0.0143 (both beyond their thresholds), every validation
  set better; 32k speed-adjusted -0.0161, AVG4 -0.011; 4k AVG4 +0.011 and clearly worse on DCLM (+0.025).
- **Equal text (D3):** 32k reads 11 % less text in the hour yet is -0.025 better at equal bytes; 16k -0.009; 4k +0.007. The gain of a
  larger vocabulary is per byte of text seen; its slower token rate works against it.
- **Evaluation window (D8, D39; overlapping windows, stride 128):**

| Vocabulary | Declared | Overlapping windows | Gap to 8k, declared | Gap to 8k, overlapping |
|---|---|---|---|---|
| 4k | 1.2032 | 1.1657 | +0.0087 | +0.0055 |
| 8k (3 seeds) | 1.1945 | 1.1602 | 0 | 0 |
| 16k (seed 0) | 1.1841 | 1.1532 | -0.0104 | -0.0070 |
| 32k | 1.1774 | 1.1489 | -0.0171 | -0.0113 |

  About one third of each gap comes from the 512-token scoring window holding more text at larger vocabularies; about two thirds is
  a better model of the text. The ordering is unchanged.
- **Test set (scored once, D47):** 16k -0.0160, 32k -0.0164, 4k +0.0094 vs the baseline: the same as on validation (test = val + ~0.005).

## Findings
- At this model size and budget, the declared 8,192 vocabulary was too small: 16k (3 seeds) and 32k (1 seed) are both better by
  ~0.016-0.017 BPB and indistinguishable from each other (difference 0.0007) -> optimal range {16k, 32k}.
- 4k is not better and probably worse (single-seed close call, +0.0087).
- Prediction check (D36): 4k worse - direction held; 16k close call - held at seed 0; 32k "inconclusive or worse" - **missed**;
  "best is 8k or 16k" - **missed** (best is the largest vocabulary tried). The compute-optimal ~5-8k of Tao et al. does not hold here.
- Candidate explanations, for the report: the token-based context holds more text at larger vocabularies (measured: ~1/3 of the gap);
  tied embeddings add parameters (E02 found 36-65M equally good, so this is a minor effect); each prediction covers more text, so the
  11 layers are spent on fewer, richer positions.
- Larger vocabularies cost more time per token than their FLOP share suggests (output layer and softmax; MFU falls with V).

## Surprises, failures and what we changed
- The best vocabulary sits at the top of the grid. No extension (65,536, the uint16 limit) was declared or run; reported as a
  limitation: the optimum is at or beyond 32k for this model and budget.
- A display bug in `summarize.py` (the "vs 8k" speed column printed +0.0 % for 4k because the 8k reference was computed after the 4k
  row). No number was affected; values were computed by hand and the fix was deferred until after the last E07 launch so the code
  fingerprint of E07's runs stayed the same (fixed in D42).
- The 4k close call was not resolved: its two extra seeds (~$2.4) could only end "inconclusive" or "worse" and change no recipe, so
  the credits went to the Phase 6 finals (D42).

## Decisions
- D36: design and rules, fixed before any run. D37: LR picks (2e-3 for all). D38: round 2 verdicts; D8 rescore brought forward.
  D39: evaluation-window check. D40: 16k BETTER with 3 seeds; optimal range {16k, 32k}. D42: 16k carried into both Phase 6
  recipes (3 seeds, lower cost per token than 32k); 4k close call dropped.
