# E07: Which vocabulary size is best in a 60-minute budget? (Phase 5, vocabulary)

## Summary
- **Question:** with the model (N49, 11 x 576, AdamW) and 60 counted minutes on one L4 fixed, does a 4,096, 16,384 or
  32,768-token vocabulary beat the declared 8,192?
- **Answer:** **16k (1.1781, 3 seeds, -0.0163) and 32k (1.1774, 1 seed, -0.0171) are both better** than 8k's 3-seed mean
  1.1945, and indistinguishable from each other (optimal range {16k, 32k}). **4k is a close call (+0.0087, 1 seed): not
  better, probably worse.** BPB falls steadily with vocabulary size: 1.2032 / 1.1945 / 1.1781 / 1.1774.
- **Why:** a larger vocabulary is slower per token (-10.5 % tokens/s at 16k, -25 % at 32k) and reads less text in the hour,
  but learns more per byte of text (32k: -0.025 at equal text read). Part of the gain is how we score: the 512-token window
  holds more text at larger vocabularies. That part is ~0.003 BPB at 16k (~21 % of its gain) and ~0.006 at 32k (~1/3); the
  rest is a better model of the text.
- **Held on every check:** machine-speed adjustment (16k -0.012, 32k -0.016), all four validation sets, the
  overlapping-window evaluation (16k -0.0129 with 3 seeds, still beyond the 3-seed threshold), and the test set, scored once
  at the end (16k -0.0160, 32k -0.0164, 4k +0.0094).
- **Tuning:** each vocabulary got the declared 4-value LR protocol; all picked 2e-3, clear of the runner-up (no close calls).
- **Consequence:** 16k (3 seeds, cheaper per token than 32k) entered both Phase 6 recipes (E08). Its gain added up with
  Muon's almost exactly (interaction +0.0011), so the declared final recipe RB = Muon + 16k is -0.041 on test.
- **Prediction missed:** we expected the best vocabulary to be 8k or 16k and 32k not to be better (Tao et al.'s fits put
  the optimum near 5-8k for our compute).
- **Figure:** none dedicated; the 16k contribution inside the recipe is in `report/figures/recipe_attribution.pdf`.
- **Cost:** 12 fifteen-minute pilots and 5 sixty-minute runs; the baseline reuses E02-N49. 8.0 counted GPU-hours, ledger
  $6.76 (GPU only) + ~$0.4 CPU data build, 5-6 Nov 2026.

## Goal
The tokenizer decides how text is cut into tokens, and with it three things at once: how much text one token covers (bytes
per token), how much of the model sits in the embedding/output layer (vocabulary x width), and how much each token costs
to process (the output layer scores every vocabulary entry). Our 8,192 vocabulary was chosen on 30 Oct by reasoning, not
measurement: GPT-2's 50k would put most of a small model's parameters and compute in the output layer. Was that reasoning
right under a time budget, where speed counts as much as learning per token?

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

## Decision rules (fixed D36, DECLARATIONS Amendments row of 5 Nov, before any run)
- **R0 data:** every run hashes its vocabulary's tokenizer and validation files; a mismatch with the verified manifests
  makes the run unusable. The variant data must pass `tools/vocab_app.py::verify` on the workspace before any run there.
- **R1 LR per vocabulary:** 5e-4, 1e-3, 2e-3, 4e-3 x 15 min, seed 0 (the grid N49 had in E01); edge rule (at most 2
  extensions per side); close call (runner-up within 2.8 sigma15 = 0.0188) -> seeds 1, 2 of both, lower 3-seed mean wins.
  No 60-minute horizon check (as E06; the baseline size had it in E01).
- **R2 per vocabulary:** 60-minute seed 0 at the pick vs the 8k 3-seed mean 1.1945; real if |diff| > 2 sigma60 sqrt(4/3) =
  0.0132; otherwise close call -> seeds 1, 2 and the 3-seed rule (1.6 sigma60 = 0.0091). If credits run short, close calls
  whose seed 0 is below 8k run first; an unrun close call is reported as inconclusive.
- **R3 replacement:** slower than 10 % below that vocabulary's own pilots (speed changes with the vocabulary by design),
  crash, warm compile, reused container or dirty code -> excluded until replaced with the same seed.
- **R4 best vocabulary:** lowest mean among those BETTER than 8k; another BETTER one within 2 standard errors -> optimal range;
  none BETTER -> 8k stays. No vocabulary outside 4k-32k (budget); a best value at the edge is reported as such.
- **Reported beside the decisions, never replacing them (D-items):** parameters, tokens/s and text/s, the equal-bytes view,
  machine-speed adjustment, four validation sets, loss per token vs per byte, the 15-minute comparison, the
  overlapping-window evaluation (D8), memory.

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

**Outcome: partly missed.** Speed: 4k held (+2.5 %); 16k (-10.5 %) and 32k (-25 %) were slower than predicted, because the
output layer and softmax cost more time than their FLOP share. BPB: 4k worse - direction held (+0.0087, close call); 16k -
held at seed 0 (-0.0104), its 3-seed mean (-0.0163) just past the range; **32k missed** (-0.0171, BETTER); **"best 8k or 16k"
missed** (16k and 32k tie, at the top of the grid). "Overlapping windows help 4k most": held (gain 0.0374 vs 0.0342 for 8k).
Kept as written.

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
| 16k (3 seeds; D43) | 1.1781 | 1.1473 | -0.0163 | **-0.0129** |
| 32k | 1.1774 | 1.1489 | -0.0171 | -0.0113 |

  The window effect is a near-constant amount per model, not a fixed share: ~0.003 BPB at 4k (in 4k's favour) and 16k
  (every 16k seed 0.0307-0.0309 sliding gain vs 0.0342 for 8k) and ~0.006 at 32k. That is ~21 % of 16k's 3-seed gain and ~1/3
  of 32k's. **Correction:** D39/D40 first wrote "about one third of each gap"; that ratio came from 16k's seed 0, whose gap was
  the smallest of its three seeds (D43). With overlapping windows 16k is still better than 8k beyond the 3-seed threshold; 32k
  (-0.0113) falls just inside the single-seed threshold (0.0132). The ordering is unchanged.
### Test set (scored once at the end, D47)
| Vocabulary | Seeds | Validation | Test | Test without contaminated documents | Test, overlapping windows | Average of four, test | Test vs baseline |
|---|---|---|---|---|---|---|---|
| 4k | 1 | 1.2032 | 1.2088 | 1.2090 | 1.1715 | 1.4063 | +0.0094 |
| 8k (E02-N49) | 3 | 1.1945 | 1.1994 | 1.1998 | 1.1655 | 1.3947 | - |
| 16k | 3 | 1.1781 | 1.1834 | 1.1843 | 1.1525 | 1.3805 | **-0.0160** |
| 32k | 1 | 1.1774 | 1.1830 | 1.1832 | 1.1540 | 1.3839 | **-0.0164** |

FineWeb-Edu test unless stated; the five E07 finals listed in `configs/frozen_configs.yaml`, each scored once on its own
vocabulary's test files after all experiments; nothing was decided from test. Test = validation + ~0.005 for every
vocabulary; removing contaminated documents moves no gap by more than 0.001; with overlapping windows the test gaps are
4k +0.0060, 16k -0.0130, 32k -0.0115, as on validation. `python code/summarize.py TEST`; files `results/test/E07-*.json`.

## Findings
- At this model size and budget, the declared 8,192 vocabulary was too small: 16k (3 seeds) and 32k (1 seed) are both better by
  ~0.016-0.017 BPB and indistinguishable from each other (difference 0.0007) -> optimal range {16k, 32k}.
- 4k is not better and probably worse (single-seed close call, +0.0087).
- Prediction check (D36): 4k worse - direction held; 16k close call - held at seed 0; 32k "inconclusive or worse" - **missed**;
  "best is 8k or 16k" - **missed** (best is the largest vocabulary tried). The compute-optimal ~5-8k of Tao et al. does not hold here.
- Candidate explanations, for the report: the token-based context holds more text at larger vocabularies (measured: ~0.003 at 16k,
  ~0.006 at 32k, i.e. ~21 % and ~1/3 of the gains);
  tied embeddings add parameters (E02 found 36-65M equally good, so this is a minor effect); each prediction covers more text, so the
  11 layers are spent on fewer, richer positions.
- Larger vocabularies cost more time per token than their FLOP share suggests (output layer and softmax; MFU falls with V:
  0.230 / 0.220 / 0.216 / 0.211 for 4k / 8k / 16k / 32k at 60 min, FLOPs counted with the output layer).
- Held on the test set: -0.016 (16k), -0.016 (32k), +0.009 (4k), the same as on validation.

### Outcome of every declared rule
| Rule | Fired? | Outcome |
|---|---|---|
| R0 data | Clean | Every run's tokenizer and validation hashes equal the verified manifests, on both workspaces used (giraffes573, joshuahanrc); variant data verified on each before use |
| R1 LR per vocabulary | Yes, all clear | 2e-3 for 4k, 16k and 32k; runner-up 0.022 / 0.023 / 0.038 behind (threshold 0.0188), so no close-call or edge pilots |
| R2 per vocabulary | Yes | 32k -0.0171 **BETTER** at seed 0; 16k -0.0104 close call -> seeds 1, 2 -> -0.0163 **BETTER** (3 seeds); 4k +0.0087 close call, seeds 1, 2 **not run** (lower priority under R2; credits went to Phase 6, D42) -> reported as inconclusive |
| R3 infrastructure | Clean | Speed vs own pilots -2.9 % to +3.9 %; all 17 runs cold compile, full budget, one attempt except one 32k pilot (5e-4), which resumed from its checkpoint after losing its machine (declared resume path, worst LR of its grid); none replaced |
| R4 best vocabulary | Yes | 16k and 32k both BETTER, 0.0007 apart (2 SE 0.0132) -> **optimal range {16k, 32k}**, at the top edge of the grid |
| D-items | Yes | Tables above: speed and text rate, equal-bytes view, machine-speed adjustment, four sets, 15-minute comparison, overlapping windows, memory (peak GPU 5.1 / 6.4 / 8.2 GB of 24) |

## Surprises, failures and what we changed
- The best vocabulary sits at the top of the grid. No extension (65,536, the uint16 limit) was declared or run; reported as a
  limitation: the optimum is at or beyond 32k for this model and budget.
- A display bug in `summarize.py` (the "vs 8k" speed column printed +0.0 % for 4k because the 8k reference was computed after the 4k
  row). No number was affected; values were computed by hand and the fix was deferred until after the last E07 launch so the code
  fingerprint of E07's runs stayed the same (fixed in D42).
- The 4k close call was not resolved: its two extra seeds (~$2.4) could only end "inconclusive" or "worse" and change no recipe, so
  the credits went to the Phase 6 finals (D42).

- **A ratio first reported from one seed had to be corrected** (window effect "~1/3" -> ~0.003 per model, ~21 % of the 3-seed
  16k gain; D43). Kept visible above rather than silently rewritten.

## How this relates to the readings
- **Tao et al. (2024), Scaling Laws with Vocabulary:** larger models deserve larger vocabularies; their compute-optimal fit
  (Approach 1) gives ~4.7k at our compute (9.6e16 FLOPs per run) and ~7.6k for our non-embedding size. We find the best
  vocabulary at 16k-32k or beyond. Their fits are in FLOPs with compute-optimal data, use their own loss normalisation and
  start at 33M non-embedding parameters; ours is a time budget, a heavily "under-trained" regime (~6 tokens per parameter)
  and a fixed 512-token context. Their remark that over-trained models want larger vocabularies points the same way only
  partly; we do not resolve the disagreement, and report it as one.
- **nanoGPT study guide:** "a fixed number of tokens can expose different amounts of preceding text ... byte normalisation
  fixes the denominator; it does not remove differences in context." D8 measures exactly this effect (~0.003-0.006 BPB) and
  shows that most of the vocabulary gain remains once it is removed. The guide's uint16 limit (ids up to 65,535) is why 32k
  is the largest vocabulary our storage format holds without a change, and why 65,536 was the natural next step not taken.
- **Smol Playbook study guide:** a vocabulary change moves embedding parameters, softmax cost and text per context together,
  so "do not attribute every difference to representation quality when throughput and capacity also changed". We report
  each part separately: parameters (non-embedding fixed), speed (tokens/s and text/s), the equal-bytes view, and the
  evaluation window.
- **Chinchilla study guide:** vary tokenization deliberately after the size study, re-measure speed and parameter counts,
  and use byte-normalised evaluation. E07 follows this order (Phase 5, after Phases 2-4) and compares only in bits per byte.
- **Kaplan et al. (2020):** non-embedding parameters as the size measure; we held them fixed (43.8M) and let the tied
  embedding grow. The total-size change (46.5M to 63.0M) lies inside E02's flat optimum (36-65M), so extra parameters alone
  are an unlikely explanation.
- **Topic page:** asks for tokenizer effects to be separated from model effects and for metrics comparable across
  tokenizers. E07 uses bits per byte throughout, scores the same documents for every vocabulary, and isolates the
  context-window part of the effect.

## Limitations and what remains uncertain
- **The optimum may lie above 32k:** the best value is at the top of the grid; 65,536 was not run (budget, and 32k already
  tied 16k). GPT-2's 50,257 tokenizer was not run either (trained on other text, a confound).
- **32k and 4k rest on one seed each.** 32k is 1.3x its threshold, and inside it under overlapping windows; 4k's close call
  was never resolved. About a dozen REAL/WORSE calls were made across the project at ~2 SE each (D41), so one false call
  among them is plausible; 16k's 3-seed result is the robust one.
- **One model size, AdamW only, one context length (512 tokens), one GPU.** The vocabulary effect was re-measured only
  inside the Muon recipe (E08), not at other sizes or contexts. A context matched in bytes rather than tokens was not run.
- **No horizon check** for the vocabularies' LR (as E06); the gap at 15 min and at 60 min had the same sign for 16k and 32k.
- **The embedding LR was tuned jointly with everything else** (one AdamW LR); E08's R1e check under Muon found 2e-3 also
  best for the embedding with 16k.
- **The tokenizers are nested BPE merges trained on one 600 MB sample**; other tokenizer algorithms or training samples
  were not compared. Rare-token under-training at 32k was not measured.
- **The 8k reference ran a day earlier on another workspace and commit**; the training path is bit-identical (tested) and
  the machine-speed adjustment does not change any verdict.

## What happened later / follow-ups
- **Phase 6 (E08):** 16k, not 32k, entered both recipes (declared tie-break, D42: 3 seeds vs 1, -10.5 % vs -25 % tokens/s,
  and the speed cost grows at width 768). Its gain given Muon was -0.0152 (alone -0.0163); Muon given 16k -0.0251 (alone
  -0.0262); **interaction +0.0011 (2 SE 0.0132): the two gains add up.** RB = Muon + 16k: -0.041 on test; RA (+ 6 x 768):
  -0.050.
- **The embedding LR question (D36 gap 4) was closed in E08 (R1e):** under Muon with the 16k vocabulary, the AdamW part
  (embeddings) was best at 2e-3 of 1e-3 / 2e-3 / 4e-3.
- **D8 for the recipes (D46):** the window effect stayed ~0.003, so ~93 % of each recipe's gain is a better model of the text.
- **Not done:** 4k seeds 1, 2; 65,536 vocabulary; a byte-matched context; vocabulary x size.

## Compute and time used
| Item | Jobs | Counted GPU time | Wall time | Ledger cost |
|---|---|---|---|---|
| LR pilots (4 values x 3 vocabularies, seed 0) | 12 | 3.0 h | 3.2 h | $2.56 |
| 60-minute runs (seed 0 x 3; 16k seeds 1, 2) | 5 | 5.0 h | 5.25 h | $4.20 |
| Vocabulary data build, giraffes573 (CPU, 4 containers) | 1 | - | ~11 min | ~$0.40 (est.) |
| Vocabulary data build, joshuahanrc (CPU; not in the ledger) | 1 | - | ~10 min | ~$0.4 (est.) |
| D8 overlapping-window rescores (8 models; not in the ledger) | 3 calls | minutes | - | cents |
| Baseline (E02-N49, reused) | 0 | - | - | - |
| **Total** | **17 GPU runs** | **8.0 h** | **8.45 h** | **$6.76 GPU + ~$0.8 CPU** |

Runs from 5 Nov 2026 19:15 (data build) to 6 Nov ~11:40, on giraffes573 (pilots, seed-0 finals; commit afb895c) and
joshuahanrc (16k seeds 1, 2). The ledger counts GPU time only; Modal billed ~1.2-1.4x in total (giraffes573 went from $10.39
to $2.98 over this phase). Modal queueing (1-2 h per round), not GPU time, set the pace.

## How to reproduce
- Data: `modal run --detach tools/vocab_app.py::build` (all three sizes + the 8,192 reproduction check), then
  `modal run tools/vocab_app.py::fetch` (first workspace only) and `modal run tools/vocab_app.py::verify` (every workspace).
  Manifests: `data/vocab/v{4096,16384,32768}/`.
- Configs: `configs/E07/V{4096,16384,32768}_lr{5e-4,1e-3,2e-3,4e-3}_15min.yaml` (seed 0) and
  `V{4096,16384,32768}_lr2e-3_60min.yaml` (seed 0; 16k also seeds 1, 2). Edge-LR and other 60-minute files exist because one
  config was written per LR before the pilots; they were not run. Baseline: `configs/E02/N49.yaml`.
- D8: `modal run modal_app.py::rescore --run-ids "..." --sets fineweb_edu --stride 128` on the workspace holding each model.
- Tables and decisions: `python code/summarize.py E07`; test: `python code/summarize.py TEST`.
- Result files: `results/runs/E07-*.json`, `results/runs/E01-N49_*.json`, `results/runs/E02-N49__s*.json`,
  `results/rescore/*__slide128.json`, `results/test/E07-*.json`. Decisions D31, D35-D40, D42-D43; DECLARATIONS Amendments row
  "Phase 5 (E07)" of 5 Nov.

## How this phase supports the report (marking criteria)
| Criterion | Evidence from E07 |
|---|---|
| Experimental design and exploration (35 %) | One factor changed (vocabulary) with the model, data documents and non-embedding size fixed; vocabularies bracket 8k on both sides; the same LR protocol 8k had; data verified by hash per workspace; a declared seed-0 rule with close-call seeds; a follow-up (D8) brought forward when the first result raised a measurement question |
| Understanding and problem formulation (25 %) | The vocabulary effect split into its parts: parameters, speed per token, text per second, learning per byte, and the evaluation window; contrasted with Tao et al. and with the reasoning behind the original 8k choice |
| Analysis and report quality (20 %) | Machine-speed adjustment, four validation sets, overlapping windows and test all agree; single-seed results and the missed prediction stated; a mis-stated ratio corrected in the open |
| Implementation and reproducibility (20 %) | Variant data built by a separate tool that cannot touch the baseline files; byte-matched learning curves; per-run data hashes; baseline training path tested bit-identical; one config per LR; every number traced to run, rescore and test files |

## Decisions
- D36: design and rules, fixed before any run. D37: LR picks (2e-3 for all). D38: round 2 verdicts; D8 rescore brought forward.
  D39: evaluation-window check (ratio corrected in D43). D40: 16k BETTER with 3 seeds; optimal range {16k, 32k}. D42: 16k carried into both Phase 6
  recipes (3 seeds, lower cost per token than 32k); 4k close call dropped. D43: D8 for all three 16k seeds. D47: test scores.
