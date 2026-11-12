# E02: Which model size is best in 60 minutes? (Phase 2, the core experiment)

## Summary
- **Question:** with 60 counted minutes on one L4, which model size predicts unseen text best, and how many tokens per
  parameter does the best size read, compared with Chinchilla's ~20?
- **Answer:** the three largest sizes tie: **N36, N49 and N65 (36-65M parameters), 3-seed means 1.1959 / 1.1945 / 1.1951
  BPB**, within 0.0015 of each other against a threshold of 0.0092. They read **4-11 tokens per parameter**. The size that
  reads ~20 per parameter in our hour (N26) is clearly worse (+0.018, five seeds). Going too small costs far more than
  going too large (N7: +0.088).
- **Held on every check:** the same ranking on the FineWeb, DCLM and Wikipedia validation sets, and on the test set,
  scored once at the end (test = validation + 0.005 for every model; N49 1.1994, N65 1.2001, N36 1.2008, N26 1.2187).
- **Why not ~20:** Chinchilla's own fitted laws predict ~20-27 tokens per parameter at our compute. A Chinchilla-style law
  fitted to our 14 complete runs puts the equal-FLOP optimum at ~56M and ~5 tokens per parameter, so equal time vs equal
  FLOPs explains only a small part of the gap (D24).
- **Consequence:** N49 (11 x 576, 48.8M, LR 2e-3) became the baseline for Phases 3-6 by the tie-break rule fixed before
  the runs (P9).
- **Figure:** `report/figures/size_vs_bpb.pdf` (`code/plots/fig_size_vs_bpb.py`): BPB vs size at 15 and 60 minutes.
- **Cost:** 17 sixty-minute runs and 8 fifteen-minute LR pilots for N49; 19.0 counted GPU-hours, ledger $15.86, 3-4 Nov 2026.


## Goal

This is the main question of the project. Every model gets exactly the same 60 minutes on the same GPU.
A small model is fast and reads a lot of text, but has little capacity to learn from it. A big model learns
more from each piece of text but reads much less in an hour. We train 6 sizes (7M to 65M parameters; a seventh, N49, was added
during the experiment, D19) for one hour each and see which ends up predicting unseen text best. Chinchilla (Reading 1) says the best model reads
about 20 tokens of text per parameter. In our hour, the 26M model happens to read ~20 tokens per parameter,
so the experiment also tests whether Chinchilla's rule holds for a time budget at this tiny scale.

We also repeat the 26M model 5 times with different seeds. That tells us how much a result moves by chance,
so we know which differences between sizes are real.

## Question
For a fixed budget of 60 counted minutes on one L4, which model size (6.9M-64.6M parameters, each at its E01
learning rate) gives the lowest final validation bits per byte on FineWeb-Edu? Where is the optimum in tokens
per parameter, compared with Chinchilla's ~20?

## Design (written before running)
| Size | Layers x width | Params | LR | Expected tokens in 60 min | Tokens/param |
|---|---|---|---|---|---|
| N7 | 6 x 256 | 7.0M | 2e-3 | ~1.48B | ~210 |
| N11 | 7 x 320 | 11.4M | 2e-3 | ~0.95B | ~83 |
| N17 | 8 x 384 | 17.5M | 4e-3 | ~0.70B | ~40 |
| N26 | 9 x 448 | 25.6M | 4e-3 | ~0.50B (measured 0.53B) | ~20 |
| N36 | 10 x 512 | 35.9M | 4e-3 | ~0.38B | ~11 |
| N65 | 12 x 640 | 64.6M | 2e-3 | ~0.23B | ~3.6 |

(Expected tokens = E01 15-min speed x ~3,570 s; N26 is measured from its 60-min run.)

- **Held fixed:** FineWeb-Edu, 8k tokenizer, context 512, 32,768 tokens/step, AdamW 0.9/0.95, weight decay
  0.1, clip 1.0, 200 warm-up steps, cosine to 10 % of peak driven by the time budget, compile on, bf16.
  Configs: `configs/E02/N*.yaml` (identical except shape and LR).
- **Runs:** 6 sizes at seed 0 (N26 seed 0 = E01-N26_lr4e-3_60min, D15) + N26 seeds 1-4 (noise
  calibration) = **9 new 60-min runs**. Real cost ~$1.0 per run (ledger estimate ~$0.83 counts the GPU only;
  Modal also bills CPU and memory, Phase 0-1 actual spend was ~1.2x the ledger), so ~$9, plus ~$2-6 for
  close calls.
- **Workspaces (revised D18):** umarmadbi: N26 seeds 1, 2 + N7 + N65; joshuahanrc: N11, N17, N36 + N26 seeds
  3, 4. Modal has been giving ~1 L4 at a time, so splitting 4/5 instead of 2/7 shortens the wall time by ~2 h,
  and spreading sizes and noise seeds over both workspaces means sigma60 includes any workspace difference.
- **Scored on all four validation splits (D18):** FineWeb-Edu (headline, decisions) plus FineWeb, DCLM and
  Wikipedia val, after training (outside the budget, ~1 min). Training is unchanged. Gives the size result on
  other text for free, and gives Phase 3 its FineWeb-Edu baseline on all four sets without rescoring.
  (The reused N26 seed-0 run has only FineWeb-Edu; it is rescored from its saved model when the rescoring
  tool for Phase 3/6 exists.)

## Prediction (written before any E02 run, D18)
The 15-min pilots (E01, best LR per size, 3-seed means) put the optimum at ~17M parameters (parabola in
log10 params: 16.6M; N17 1.3057 lowest). 60 min is 4x the compute for every size. If the optimal size grows
as compute^a, Chinchilla (a ~0.5) predicts ~34M at 60 min and Kaplan (a ~0.73) ~46M. Equal time is not equal
FLOPs: small models use the L4 less efficiently, which should pull the time-optimum somewhat smaller. **Predicted
60-min optimum: ~25-35M (N26-N36), i.e. ~10-20 tokens/param.** P7 tests this.

**Outcome: missed.** The optimum is 36-65M (3.8-11 tokens/param). The sentence on equal time vs equal FLOPs has the
direction wrong: bigger models use the L4 more efficiently, so equal time gives them more FLOPs, which favours them
(D19). Kept as written.

## Decision rules (D17, revised D18 before any new E02 run)
- **P1 Best size**, in three steps (final validation BPB on the full FineWeb-Edu split, final model):
  1. **Seed 0** ranking of all sizes. sigma60 = sample sd of N26 seeds 0-4 (4 df).
  2. **Close-call set** = every size within 2.8 sigma60 of the best seed-0 size, at most the 3 best
     (D18: was "best and runner-up only"; with a flat optimum the 3rd size can be within noise too, and
     picking the runner-up by one seed is the same single-seed luck E01 showed at N11). Plus any size
     flagged by P10. If the set is only the best size: decided at seed 0. Otherwise every size in it gets
     seeds 1 and 2 (N26 already has them).
  3. **3-seed means** (seeds 0-2 for every size, N26 included; N26's 5-seed mean also reported). Winner =
     lowest mean; real only if it beats every other member by > 1.6 x pooled sigma60, where pooled sigma60
     uses every size with >= 2 seeds (N26's 4 df + 2 df per close-call size; D18, same rule as D11 in E01:
     the threshold uses the best noise estimate available at the decision). Otherwise the result is an
     **optimal range** = the winner and every member within 1.6 x pooled sigma60 of it.
- **P2 Edge rule:** if the final best is N7 or N65, run one size beyond the edge (`configs/E02/N4.yaml` 3.9M
  or `N92.yaml` 91.6M), at its neighbour's LR (not tuned; disclosed).
- **P2b Refinement (D18):** if the P3 optimum falls between N36 and N65 (the widest gap in the grid, x1.8),
  add N49 (11 x 576, 48.8M): LR tuned with the declared protocol (`configs/E01/N49_lr{5e-4,1e-3,2e-3,4e-3}`,
  4 x 15 min, ~$1), then `configs/E02/N49.yaml` seed 0 at the pick (it joins P1 like any size). ~$2.
- **P3 Where is the optimum?** Parabola of BPB against log10(total parameters) over all sizes (Chinchilla
  "Approach 2", but along an equal-TIME line), weighted by the number of seeds, on each size's mean over all
  its seeds. Reported with the leave-one-size-out range (D18) and the tokens/param at the optimum. Diagnostic.
- **P4 Equal time is not equal FLOPs:** report each size's training FLOPs (6N + attention, `flops_per_token`)
  and MFU. Discussed, not corrected for: the declared budget is time.
- **P5:** decisions use final BPB only (E01: curves of different settings cross in the last minutes).
- **P6 Depth vs width** is designed after P1, at the P9 size, with its own rules recorded before its runs.
- **P7 How the optimum moves with compute (D18, free):** the E01 15-min runs at each size's picked LR are
  complete runs (their cosine schedule finishes at 15 min), so they are valid final points at 1/4 of the
  time. Fit the same parabola at 15 and 60 min; exponent a = log(N_opt,60 / N_opt,15) / log 4, compared with
  Chinchilla ~0.5 and Kaplan ~0.73. Caveats stated: two points only; warm-up and compile are a larger share
  of 15-min runs; LRs were tuned at 15 min.
- **P8 How much of the seed noise is machine speed (D18, free):** regress the N26 seeds' BPB on
  ln(tokens processed). Diagnostic only; never used to adjust a decision.
- **P9 Baseline size for Phases 3-6 (D18, fixed now so it is not chosen after seeing results):** the P1
  winner if it is real. If P1 gives an optimal range that includes N26, N26 is the baseline (statistically
  indistinguishable from the best, and its 5 seeds are reused as the FineWeb-Edu arm of Phase 3 and the
  baseline arm of Phase 6, saving ~3-5 runs). Otherwise the lowest-mean size.
- **P10 Instability flag (D18):** a run with > 5 % clipped steps or a training-loss jump > 0.1 between
  consecutive 30-s log intervals after warm-up (healthy 60-min N26 runs: 0.16 % clipped, max jump 0.013) is
  flagged, kept (it is a property of the config, not an infrastructure fault), and its size gets seeds 1 and 2
  so one unstable run cannot drop a size from the comparison. Reason: N36 at its pick (4e-3) had a 15-min seed
  sd of 0.0105 vs ~0.003 for safe LRs, and N36 is next to the predicted optimum.

## Gaps and confounders checked
- **Repetition:** none. N7 reads ~1.5B of 2.45B FineWeb-Edu train tokens (0.6 passes).
- **Unequal tuning:** every size has an E01-tuned LR; within each size 2e-3 and 4e-3 were tied, so an LR
  pick costs at most a few thousandths of BPB (stated as a limitation).
- **Noise differs by size (D18):** sigma60 is measured at N26 only. Sizes at a riskier LR (N36 at 4e-3,
  N11) may be noisier. Handled by P1 step 2 (3 seeds for every contender), P10, and pooled sigma.
- **sigma60 has only 4 df (D18):** its 95 % range is roughly 0.6x-2.9x the estimate, so the seed-0 threshold
  is rough. That is why the seed-0 step only selects contenders and the final call uses 3-seed means and the
  pooled estimate.
- **Parameter convention (D18):** with an 8k vocabulary the embedding is 30 % of N7 but 8 % of N65, so
  tokens/param depends on what is counted (N7: ~210 total vs ~320 non-embedding). Chinchilla counts all
  parameters, Kaplan non-embedding; Pearce & Song (2024) trace part of the Kaplan-Chinchilla disagreement to
  this. Both are reported; total is the headline because Chinchilla's ~20 uses it.
- **Hardware noise:** L4s run power-capped at machine-specific clocks (+-2-4 % tokens). It enters sigma60
  through the 5 seeds (P8 quantifies it); tokens processed and SM clock are reported for every run.
- **Slow-machine rule (D18 made it checkable):** `summarize.py E02` compares each run's steady speed with its
  size's E01 pilot median (steady speed, because compile is ~3 % of a 15-min run but ~1 % of a 60-min run)
  and flags > 10 % below -> replacement with `--fresh`, same seed. Also flags warm compile, reused container,
  preemption, dirty code, non-L4, short budget. Before D18 none of these checks ran for E02.
- **Per-step GPU syncs (`loss.item()` per micro-step, grad-norm `.item()`):** fixed overhead per step that
  costs small, fast models a larger share of their time. Part of why small models have lower MFU. Same code
  for all sizes and for E01; changing it now would break comparability with E01 and the reused run. Stated as
  a limitation; a candidate for the Phase 4 efficiency experiment.
- **Warm-up is 200 steps**, i.e. ~0.4 % of N7's hour but ~2.8 % of N65's. Same rule for all; limitation.
- **Compile warm-up** is counted and differs by size (18-40 s, < 1.2 % of the hour). Part of the declared budget.
- **Preemption:** a resumed run pays a second compile (~30 s). Kept, flagged (attempts > 1), disclosed.
- **Mixed commits:** N26 seed 0 ran from commit 9156019; the others will run from a later commit. The
  difference is logging, launch, memory request, summary and the extra validation sets (scored after
  training); the training path is identical. `summarize.py E02` prints commit and code hash per run.
- **Selection of N26 seed 0:** chosen from 15-min results before its 60-min score (D15).
- **Test set:** not touched.

## Run order
1. Both workspaces: `verify_data` PASSED (13/13), gpu_check OK (3 Nov).
2. Commit D18 (code, configs). Launch, each workspace in its own terminal, `--detach`:
   - umarmadbi: `configs/E02/N7.yaml,configs/E02/N65.yaml` seed 0; `configs/E02/N26.yaml` seeds 1,2
   - joshuahanrc: `configs/E02/N11.yaml,configs/E02/N17.yaml,configs/E02/N36.yaml` seed 0;
     `configs/E02/N26.yaml` seeds 3,4
3. `python code/summarize.py E02`: resolve any infrastructure flag first, then P1 step 2 (seeds 1,2 for the
   close-call set), P2/P2b if triggered, P1 step 3, P3, P7-P9.

## Runs
### Batch 1 (2026-11-03 14:30-~16:00, commit 90f6087)
| Run | Workspace | Final BPB (FineWeb-Edu val) | FineWeb / DCLM / Wikipedia val | Tokens | Tok/param (total) | SM clock | Speed vs pilots |
|---|---|---|---|---|---|---|---|
| N7 s0 | umarmadbi | 1.2824 | 1.4701 / 1.6276 / 1.5552 | 1,534M | 221 | 1,035 | +1.4 % |
| N11 s0 | joshuahanrc | 1.2426 | 1.4325 / 1.5861 / 1.5182 | 989M | 87 | 968 | +0.4 % |
| N17 s0 | joshuahanrc | 1.2296 | 1.4187 / 1.5783 / 1.5039 | 763M | 44 | 1,042 | +5.6 % |
| N26 s0 (E01 horizon run, D15) | umarmadbi | 1.2058 | 1.3969 / 1.5525 / 1.4742 (rescored from the saved model, D42) | 528M | 21 | 945 | +1.8 % |
| N26 s1 | umarmadbi | 1.2193 | 1.4092 / 1.5646 / 1.4904 | 488M | 19 | 810 | -5.8 % |
| N26 s2 | umarmadbi | 1.2164 | 1.4063 / 1.5622 / 1.4882 | 501M | 20 | 848 | -3.3 % |
| N26 s3 | joshuahanrc | 1.2127 | 1.4032 / 1.5576 / 1.4896 | 519M | 20 | 870 | -0.6 % |
| N26 s4 | joshuahanrc | 1.2074 | 1.3976 / 1.5558 / 1.4783 | 549M | 21 | 1,058 | +6.0 % |
| **N36 s0** | joshuahanrc | **1.1888** | 1.3802 / 1.5358 / 1.4571 | 413M | 11.5 | 922 | +2.6 % |
| **N65 s0** | umarmadbi | **1.1899** | 1.3802 / 1.5355 / 1.4533 | 245M | 3.8 | 960 | +3.6 % |

All: 1 attempt, cold compile (first step 21-39 s), one run per container, clipped steps 0.08-0.32 %, peak host RAM 9.6 GB.
Cost ~$0.81-0.85 each in the ledger (~$1.0 real).

### Batch 2 (2026-11-03 16:39-~18:15, commit 50ec597, joshuahanrc): close-call seeds
| Run | Final BPB | FineWeb-Edu tokens | SM clock | Speed vs pilots |
|---|---|---|---|---|
| N36 s1 | 1.1963 | 397M | 870 | -0.8 % |
| N36 s2 | 1.2026 | 397M | 840 | -0.6 % |
| N65 s1 | 1.1923 | 246M | 938 | +3.2 % |
| N65 s2 | 1.2030 | 238M | 825 | +0.3 % |

All clean (1 attempt, cold compile, no flags). N49 LR pilots (E01 configs, 15 min, umarmadbi): 5e-4 1.4061,
1e-3 1.3774, 2e-3 1.3577, 4e-3 1.3710 (close call 2e-3 vs 4e-3 -> seeds 1, 2 of both, D20).

### Batch 3 (2026-11-04 02:48-~06:05, commit 0b7dde4, jinkyungjeon): N49 (LR 2e-3, picked by 3-seed pilots)
| Run | Final BPB | FineWeb / DCLM / Wikipedia val | Tokens | SM clock | Speed vs pilots |
|---|---|---|---|---|---|
| N49 s0 | 1.1936 | 1.3845 / 1.5413 / 1.4648 | 288M | 855 | -3.8 % |
| N49 s1 | 1.1961 | 1.3863 / 1.5422 / 1.4607 | 292M | 870 | -2.4 % |
| N49 s2 | 1.1937 | 1.3841 / 1.5376 / 1.4593 | 292M | 930 | -2.4 % |

## Results
### After batch 1 (seed 0), D19
- **The bigger models win.** BPB falls steadily from N7 to N36 and is flat from N36 to N65 (0.0011 apart). Same
  ranking on all four validation sets.
- **Noise:** sigma60 = 0.0058 (N26, 5 seeds). 84 % of it follows machine speed (P8). Threshold 2.8 sigma60 = 0.0161.
- **P1 step 2:** close-call set {N36, N65} -> seeds 1, 2 each. N26 is just outside on seed 0 (+0.0170) and well
  behind on its 5-seed mean (+0.0235).
- **P3 (corrected, D19):** whole-range parabola optimum ~103M, beyond the grid, unstable (leave-one-out 71M to
  >10 G): not informative. Post-hoc local fit through N26/N36/N65: ~48M, ~7 tokens/param.
- **Prediction (25-35M) missed on the low side;** part of the reasoning had the MFU effect backwards (D19).
- **Next:** N36 and N65 seeds 1, 2; N49 follow-up (declared D19 before those seeds).

### After batch 2 (D20)
- **3-seed means: N36 1.1959, N65 1.1951** (gap 0.0008 vs threshold 0.0102, pooled sigma 0.0064 on 8 df):
  **optimal range {N36, N65}**, provisional until N49 joins.
- Winner's curse: both sizes' seed-0 runs were their best (fast machines); later seeds 0.004-0.014 worse.
- N92 not run (N65 lowest mean but not an outright win; D19 rule). The plateau may extend above 65M: limitation.
- P9 provisional baseline: N65.

### Final (D22)
| Size | Params | Tokens/param (total / non-emb) | Seeds | Mean BPB (seeds 0-2) | FLOPs (PFLOP) | MFU |
|---|---|---|---|---|---|---|
| N7 | 7.0M | 221 / 325 | 1 | 1.2824 | 77 | 0.18 |
| N11 | 11.4M | 87 / 115 | 1 | 1.2426 | 80 | 0.18 |
| N17 | 17.5M | 44 / 54 | 1 | 1.2296 | 94 | 0.22 |
| N26 | 25.6M | 20 / 24 | 5 | 1.2138 (5-seed 1.2123) | 91 | 0.21 |
| **N36** | 35.9M | 11 / 13 | 3 | **1.1959** | 99 | 0.23 |
| **N49** | 48.8M | 6.0 / 6.6 | 3 | **1.1945** | 96 | 0.22 |
| **N65** | 64.6M | 3.8 / 4.1 | 3 | **1.1951** | 105 | 0.24 |

P1: optimal range {N36, N49, N65} (gaps 0.0006-0.0015 vs threshold 0.0092; pooled sigma60 0.0057, 10 df).
P9 baseline: N49. P3 whole-range fit ~70M (LOO 57-96M); post-hoc local fit ~51M, ~5.5 tokens/param.

### The same ranking on other text (validation, means over seeds 0-2)
| Size | FineWeb-Edu | FineWeb | DCLM | Wikipedia | Average of four |
|---|---|---|---|---|---|
| N7 | 1.2824 | 1.4701 | 1.6276 | 1.5552 | 1.4838 |
| N11 | 1.2426 | 1.4325 | 1.5861 | 1.5182 | 1.4449 |
| N17 | 1.2296 | 1.4187 | 1.5783 | 1.5039 | 1.4326 |
| N26 | 1.2138 | 1.4041 | 1.5598 | 1.4843 | 1.4155 |
| N36 | 1.1959 | 1.3863 | 1.5405 | 1.4657 | 1.3971 |
| N49 | 1.1945 | 1.3850 | 1.5404 | 1.4616 | 1.3954 |
| N65 | 1.1951 | 1.3857 | 1.5413 | 1.4611 | 1.3958 |

All models were trained on FineWeb-Edu only. On every set the three largest sizes tie (within 0.002 on the average of
four) and each smaller size is worse. N26 seed 0 was scored on the three extra sets after training (D42).

### Test set (scored once at the end, D47)
| Size | Seeds | Validation | Test | Test - validation | Test without contaminated documents |
|---|---|---|---|---|---|
| N7 | 1 | 1.2824 | 1.2871 | +0.0047 | 1.2871 |
| N11 | 1 | 1.2426 | 1.2484 | +0.0058 | 1.2485 |
| N17 | 1 | 1.2296 | 1.2346 | +0.0050 | 1.2347 |
| N26 | 3 (seeds 0-2) | 1.2138 | 1.2187 | +0.0049 | 1.2190 |
| N36 | 3 | 1.1959 | 1.2008 | +0.0049 | 1.2011 |
| **N49** | 3 | **1.1945** | **1.1994** | +0.0049 | 1.1998 |
| N65 | 3 | 1.1951 | 1.2001 | +0.0050 | 1.2004 |

FineWeb-Edu test, the 17 Phase 2 models listed in `configs/frozen_configs.yaml` ("Phase 2 size"), scored once after all
experiments were finished. Test is higher than validation by +0.0050 +- 0.0003 for every run (range +0.0045 to +0.0058),
so the ordering of sizes is unchanged (rank correlation over the 17 runs 0.998) and the optimal range and the N26 gap
(+0.019 on test) reproduce. Removing test documents that overlap the training data changes nothing (<= +0.0005).
`python code/summarize.py TEST`.

## Findings
1. **In a 60-minute L4 budget the best models are 36-65M parameters trained on only ~4-11 tokens per parameter**,
   far below Chinchilla's ~20. The 20-tokens/param model (N26) is clearly worse (+0.018 BPB, ~3 sigma).
2. **The optimum is flat**: three sizes over a x1.8 range agree within 0.0015 BPB. Going too small costs much more
   (N7: +0.088) than going too large within the grid.
3. **The optimum grows fast with budget:** ~17M at 15 min -> ~50-70M at 60 min (N_opt ~ time^0.8-1.0).
4. **Equal time is not equal compute:** bigger models use the L4 better (MFU 0.18 -> 0.24), and machine-to-machine
   speed differences explain most of the seed noise (P8: 84 % for N26).
5. **Seed-0 winner's curse:** the first runs of N36 and N65 were their best; 3 seeds were needed to see the tie.
6. **Our prediction (25-35M) was too small**, partly because we reasoned the MFU effect in the wrong direction (D19).

### Post-hoc: why ~5 tokens/param and not ~20? (D24, diagnostic only)
`python code/fit_scaling_law.py` fits L = E + A/N^a + B/D^b to the 14 complete runs (7 sizes x 15 and 60 min):
a = 0.16, b = 0.84, max residual 0.013 BPB. Equal-FLOP optimum at our 60-min budget: ~56M, ~4.8 tokens/param
(leave-one-out 40-60M, 4.2-9.6). Chinchilla's published laws at the same budget: ~27M / ~20 tokens/param
(Hoffmann et al.) and ~24M / ~27 (Besiroglu et al.). So the gap is not mainly equal time vs equal FLOPs; it is
how loss scales in our regime (data, recipe, tiny compute). See D24 for caveats.

### Outcome of every declared rule
| Rule | Fired? | Outcome |
|---|---|---|
| P1 best size | Yes, all three steps | Seed 0: N36 best; close-call set {N36, N65, N49}; 3-seed means tied -> **optimal range {N36, N49, N65}** |
| P2 edge sizes (N4, N92) | No | Lowest mean is N49, an interior size. N92 not run, so the plateau above 65M is untested |
| P2b refinement (N49) | Replaced (D19) | Its trigger depended on the buggy fit; N49 was declared unconditionally before the deciding seeds and run with its own tuned LR |
| P3 parabola | Diagnostic | Whole range ~70M (LOO 57-96M), beyond the grid; local fit through N36/N49/N65 ~51M (post hoc) |
| P4 FLOPs | Reported | 77-105 PFLOP; MFU 0.18 (N7) to 0.24 (N65) |
| P5 final BPB only | Applied | All decisions on final validation BPB |
| P6 depth vs width | Moved to E06 | At N49's size: 6 x 768 better via speed, 19 x 448 worse |
| P7 optimum vs budget | Diagnostic | ~17M at 15 min -> ~50-70M at 60 min (time^0.8-1.0; Chinchilla 0.5, Kaplan 0.73) |
| P8 machine speed | Diagnostic | N26: r^2 = 0.84, slope -0.116 per ln(tokens); pooled within N26/N36/N49/N65 (14 runs): -0.145 +- 0.035, r^2 = 0.66 |
| P9 baseline size | Yes | N49 (lowest mean in the range; N26 not in the range) |
| P10 instability | Never | Clipped steps 0.08-0.32 % in all 17 runs; no loss jumps |

## Surprises, failures and what we changed
(Compiled from D18, D19 and the lab log of 3-4 Nov.)
- **The summary script did not apply the declared rule** (found before any new E02 run, D18): it ranked sizes by the mean over
  all seeds, so N26's five seeds were compared with other sizes' single seeds. Fixed to the declared order (seed 0 -> close-call
  set -> 3-seed means), and the infrastructure checks (warm compile, reused container, slow machine) extended to E02.
- **The diagnostic parabola fit was wrong** (D19): the normal equations were ordered (1, x, x^2) instead of (x^2, x, 1), so it
  printed an optimum of 20.7M while the two best sizes were 36M and 65M. Fixed. Because the declared P2b trigger depended on
  that fit, N49 was declared as a follow-up before the N36/N65 close-call seeds were known, and run regardless of their outcome.
- **Seed-0 winner's curse:** the first runs of N36 and N65 were their best; only 3 seeds showed the tie.
- **Noise was larger than assumed:** sigma60 (0.0058) is about the same as sigma15, not smaller; 84 % of N26's seed variance
  follows machine speed (seeds on slow L4s at 810-848 MHz vs 1,058 MHz).
- **Prediction missed** (25-35M): the optimum is 36-65M, partly because we reasoned the MFU effect in the wrong direction.
- **Recording problems:** both launch terminals lost their connection (runs continued on Modal and were recovered); one
  `recover` failed writing the ledger because the file was open in another program, and was rerun with every row intact.

## How this relates to the readings
- **Hoffmann et al. (2022, Chinchilla):** our design is their "Approach 2" (a parabola through final losses of several
  sizes at one budget), except that the budget is equal *time* rather than equal FLOPs. Their ~20 tokens per parameter is
  not reproduced: their own fitted law predicts ~27M and ~20 tokens per parameter at our compute, where we measure 36-65M
  and 4-11. Their smallest budgets were ~60x ours, so our regime is an extrapolation of their fit.
- **Kaplan et al. (2020):** larger models are more sample-efficient, and the compute-optimal size grows faster with
  compute (exponent ~0.73) than Chinchilla found (~0.5). Our 15-to-60-minute shift (exponent ~0.8-1.0) and our preference
  for large models are closer to Kaplan's picture. Kaplan counted non-embedding parameters; we report both conventions,
  which matters here because the embedding is 30 % of N7 and 8 % of N65.
- **Pearce & Song (2024), Porian et al. (2024) and Besiroglu et al. (2024):** the Kaplan/Chinchilla disagreement comes
  partly from counting non-embedding parameters at small scale (Pearce & Song) and from last-layer FLOPs, warm-up length
  and per-size optimiser tuning (Porian et al.); Besiroglu et al. refit Chinchilla's data and get ~27 tokens per
  parameter at our compute. Both support the point that small-scale optima are
  sensitive to setup, which is why we tuned the learning rate per size (E01) and report the parameter convention.
- **Smol Training Playbook / SmolLM2:** small production models are trained far beyond Chinchilla (trillions of tokens)
  because inference cost matters. Our objective has no inference cost and a hard training budget, and it points the other
  way: fewer tokens per parameter.
- **DeepSeek LLM (2024):** reports that higher-quality data shifts the optimal allocation toward larger models; FineWeb-Edu
  is a heavily filtered corpus, one candidate reason for our low ratio (not tested here).

## Limitations and what remains uncertain
- **One model family, one GPU type, one budget length.** The time-optimum depends on how efficiently the L4 runs each
  size (MFU 0.18-0.24); another GPU would move it.
- **The optimum is flat and only bounded from below.** Three sizes tie; N92 was not run, so sizes above 65M may be as good.
  The location of the optimum (P3, D24) is loosely determined.
- **Small sizes have one seed each** (N7, N11, N17). Their gaps to the optimum (>= 0.035) are several times the noise, so
  the ranking is safe, but their exact values carry +- ~0.006.
- **Learning rates were tuned at 15 minutes** (E01). The 60-minute check was done for N26 only; elsewhere 2e-3 and 4e-3
  were within noise, so a mis-pick costs at most a few thousandths.
- **Fixed settings across sizes:** batch (32,768 tokens), 200 warm-up steps (0.4-2.8 % of the hour), AdamW betas, weight
  decay, per-step GPU synchronisation overhead. Each was the same for every size but may suit some sizes better.
- **Noise:** sigma60 rests on 10 degrees of freedom; most of it is machine speed, which the declared metric does not
  correct for (P8 is reported only).
- **The scaling-law fit (D24) is post hoc,** with 14 points for 5 parameters and only two budgets per size.
- **BPB measures text prediction only,** not downstream ability (DECLARATIONS section 5).

## Follow-ups this experiment led to
- **E04 F2:** with unique data limited to 36M tokens, does a smaller model become best? No: N65 was still best (D27).
- **E06 (P6):** shape at N49's size: 6 x 768 beat 11 x 576 by 0.018 through 16 % higher speed, consistent with the
  finding here that GPU efficiency drives the time-optimum.
- **E03-E08** all use N49 as the baseline; the final recipe (E08) improves on it by 0.041 BPB on test.

## Compute and time used
| Item | Jobs | Counted GPU time | Wall time | Ledger cost |
|---|---|---|---|---|
| E02 60-minute runs (incl. N26 seeds 1-4) | 16 | 16.0 h | 16.6 h | $13.28 |
| N26 seed 0 (E01 horizon run, reused, D15) | 1 | 1.0 h | 1.0 h | $0.82 |
| N49 learning-rate pilots (E01 configs, 15 min) | 8 | 2.0 h | 2.2 h | $1.76 |
| **Total** | **25** | **19.0 h** | **19.8 h** | **$15.86** |

Ran 3 Nov 14:30 to 4 Nov ~06:05 on three Modal workspaces (umarmadbi, joshuahanrc, jinkyungjeon). The ledger counts GPU
time only; Modal also bills CPU and memory (about 1.2x in total). No run was replaced; `failed` ledger rows from dropped
laptop connections are annotated and carry no cost. Two runs appear twice in the ledger with identical values
(`E02-N65__s2`, `E01-N49_lr4e-3__s0`, recorded twice on recovery); they are counted once here.

## How to reproduce
- Configs: `configs/E02/N*.yaml` (N4 and N92 written, never run) and `configs/E01/N26_lr4e-3_60min.yaml` (N26 seed 0).
- Runs: `modal run modal_app.py::launch --configs configs/E02/N49.yaml --seeds 0,1,2 --kind final` (see `docs/REPRODUCE.md`).
- Tables and decisions: `python code/summarize.py E02`; test set: `python code/summarize.py TEST`.
- Scaling-law fit (D24): `python code/fit_scaling_law.py`. Figure: `python code/plots/fig_size_vs_bpb.py`.
- Result files: `results/runs/E02-*.json`, `results/runs/E01-N26_lr4e-3_60min__s0.json`, `results/test/E02-*.json`,
  `results/rescore/E01-N26_lr4e-3_60min__s0__full.json`.
