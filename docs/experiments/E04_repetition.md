# E04: How much does repeating the same text hurt? (Phase 3b, repetition)


## Goal
Our models never saw the same text twice (an hour of N49 reads ~290M of 2.45B tokens). With less data, the same text has
to be repeated. Muennighoff et al. found that repeating up to ~4 times is almost as good as new data, and that the value
of further repeats fades. We give N49 its usual hour but only a small random slice of FineWeb-Edu, sized so that
the hour reads it ~2, 4, 8, 16 or 32 times, and compare with the unique-data N49 runs.

Beyond "where does it start to hurt", we also translate each result into **how much new data it was worth**
(effective data), which is the quantity Muennighoff et al. model, so our numbers can be compared with theirs.

## Question
At N49 and 60 minutes, from how many passes over the same data does validation BPB become measurably worse than
with unique data, and how fast does it degrade?

## Prediction (written before any E04 run)
2 and 4 passes: not distinguishable from unique data. 8: slightly worse (~0.005-0.01). 16: clearly worse.
32: much worse, with the validation curve flattening or rising before the end (overfitting).

## Design
- **What varies:** size of the training subset: 145M, 72.5M, 36.25M, 18.1M, 9.06M tokens (= 290M / k for
  k = 2, 4, 8, 16, 32). `configs/E04/N49_rep{k}.yaml`, seed 0. 5 runs (~$5), plus up to 6 follow-up seeds (R2).
- **Subsets are random and nested (D23):** `train_subset_tokens` picks whole 513-token windows by a fixed
  permutation (key 1234), so each subset is a uniform sample of the whole train split and every smaller one is
  inside every larger one. (The old `train_tokens_limit` took the first N tokens in file order, i.e. mostly the
  first source file, which is also where validation documents were sampled from: not a fair subset. Not used.)
- **Unique-data reference:** E02-N49 seeds 0-2 (same config without a subset; 0.12 passes).
- **Held fixed:** everything else as in E02 N49 (LR 2e-3, schedule, budget, FineWeb-Edu).
- **Workspace:** joshuahanrc (all 5, and the follow-up seeds).

## Decision rules (fixed D23; R2 revised and R5-R6 added in D25, before any E04 run)
- **R1:** a k-pass run is **worse** than unique data if its final FineWeb-Edu val BPB exceeds the reference 3-seed
  mean by more than 2 x sigma60 x sqrt(1 + 1/3) = 0.0132 (sigma60 = 0.0057 pooled, E02; single run vs a 3-seed
  mean). Otherwise "not distinguishable".
- **R2 Boundary (revised D25, so it works if the seed-0 results are not monotonic in k):** k* = the **smallest** k
  called worse at seed 0. The bracket {next smaller k, k*} gets seeds 1, 2 (only k* if k* = 2). With 3 seeds, k is
  worse if its mean exceeds the reference by > 1.6 sigma60 = 0.0091. If both bracket ends are worse, the boundary
  moves **down** one step (that k gets seeds 1, 2); if k* is not worse, it moves **up** one step; at most one move.
  Result: "repetition becomes measurably worse between k_safe and k_worse passes", using 3-seed verdicts where they
  exist. If no k is worse at seed 0: result "> 32 passes", no follow-ups. A larger k that is "not distinguishable"
  after a smaller k was worse is reported as noise. Seeds use the same subset (fixed key), so they measure training
  noise, not which slice was drawn. Cost: up to 6 extra runs (~$6), typically 4.
- **R3 Reported, not decisive:** AVG4 over the four validation sets; passes actually made (they vary with machine
  speed); the gap between the best and final point of the 1 MB validation curve (overfitting sign).
- **R4** Infrastructure checks as in E02.
- **R5 Diagnostics (D25, not decisive):** (a) each delta adjusted to the reference's tokens processed (machine speed,
  slope -0.136 per unit ln tokens from E02); (b) the **train-validation gap**: final FineWeb-Edu val loss minus the
  mean training loss over the last 5 % of the budget, relative to the reference's gap. With unique data both losses
  are on unseen text; with repeated data a growing gap is memorisation, visible before the validation loss itself
  turns up.
- **R6 Effective data (D25, diagnostic):** `code/fit_repetition.py` inverts our unique-data loss law
  (`fit_scaling_law.py`, D24) at N = N49 to turn each run's BPB into the tokens of new data that would give the same
  loss (D'), and fits Muennighoff et al.'s D' = U + U R* (1 - exp(-R/R*)) (R = passes - 1) to get R*, compared with
  their R*_D ~ 15.4. The calibration line (the three unique-data runs should come out at ~1x) is printed with it;
  with sigma60 = 0.0057 one seed moves D' by roughly +-10-15 %, so R* is a rough number.

## Follow-ups (declared D26, after the seed-0 results and before any follow-up run)
Seed 0 showed far more damage than Muennighoff et al. report (4 passes +0.025, 8 passes +0.099). Two possible causes in
our setup, each tested with one cheap run per point (seed 0; vs E04-N49_rep8 seed 0, single vs single: real if
|diff| > 2.8 sigma60 = 0.0161):
- **F2 Model too large for the data (also: "where should the hour go when unique data is limited?").** N17, N26, N65
  at their E01 LRs on the SAME 36.25M-token subset as N49 k = 8 (`configs/E04/{N17,N26,N65}_U36M.yaml`). In 60 min they
  read it ~21, ~14, ~7 times. Also reported: each size's "cost of the limit" (vs its own unique-data E02 mean), passes,
  train-val gap, where its validation curve bottoms out. Muennighoff et al.: excess parameters lose value faster than
  repeated data (R*_N ~5.3 < R*_D ~15.4), so a smaller model should do better with limited data.
  **Prediction:** N17 and N26 better than N49 (less memorisation despite more passes); N65 worse.
- **F3 Exact repetition of identical samples.** Our loader shows the same 513-token windows every pass (same
  context/target pairs); common pipelines re-cut the text each epoch. `configs/E04/N49_rep8_shift.yaml`: same subset
  and model, window boundaries move every pass (`train_subset_shift`, `ShiftedSubsetLoader`; the seams between subset
  windows are marked with EOT, adding 0.2 % tokens). **Prediction:** modestly better (0.01-0.03), far from closing the
  +0.099 gap.
- Workspaces: giraffes573 N17, N26; jinkyungjeon N65, F3. ~$4.
- Not tested (limitations): regularisation (dropout, weight decay), LR re-tuned for repeated data.

## Gaps and confounders checked
- **Passes depend on machine speed** (+-4 %): the actual passes are reported; k labels are nominal. R5a adjusts.
- **One random subset per k:** a different slice could score slightly differently; at 9M tokens that slice is
  still ~17,700 windows from all over the train split. Stated.
- **Settings tuned for unique data:** LR, weight decay (0.1) and no dropout were not re-tuned for repeated data;
  regularisation could reduce the damage at high k. Limitation; Muennighoff et al. also kept settings fixed.
- **"Not distinguishable" is not "equal" (D25):** the single-seed threshold (0.0132, ~1.1 % of BPB) is coarser than
  the < 0.5 % loss differences Muennighoff et al. report for <= 4 epochs; we report each delta with its noise and
  the effective-data ratio, not only the label.
- **Model much larger than the data at high k:** at k = 32 N49 (48.8M params) sees 9M unique tokens (0.19 tokens per
  parameter), a regime where memorisation is expected; it is the end point of the dose-response, not a recipe.
- **Windows, not documents:** the subset is drawn as 513-token windows, exactly the units the baseline trains on.
- **Evaluation:** FineWeb-Edu val (same distribution as the training text); val documents are not in train (D02).
- **Mixed commits:** reference from 0b7dde4; the subset path only activates when `train_subset_tokens` > 0 (tested).
- **Not tested:** repetition on another corpus; the data-constrained allocation question (smaller model vs more
  repeats at fixed unique data). Limitations.

## Run order
1. Commit D23 + D25. joshuahanrc: `modal profile current`, then
   `configs/E04/N49_rep2.yaml,...,N49_rep32.yaml` seed 0 (`--kind final`).
2. `python code/summarize.py E04` -> R2 bracket seeds; again after they land; `python code/fit_repetition.py`.

## Runs
Commit 4fce312 (D23 + D25), joshuahanrc, launched 11:56; starts 12:27 (k=2), 12:29 (4), 13:42 (8), 13:43 (16), 13:52 (32).
Launch terminal lost its connection (5 'failed' ledger rows, to annotate); all completed (1 attempt, cold compile, one
run per container) and were recovered ~15:00.
R2 bracket seeds (k = 2, 4 x s1, s2): commands given 15:05, joshuahanrc (launch time to be logged).

## Results
### Seed 0 (D26)
| k | Unique tokens | Passes | FWE val | vs unique (1.1945) | Speed-adjusted | AVG4 diff | Train-val gap vs ref | Best 1 MB curve point | Verdict (R1) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 145.0M | 2.00 | 1.2044 | +0.0099 | +0.0099 | +0.0129 | +0.02 | end of run | not distinguishable |
| 4 | 72.5M | 4.14 | 1.2190 | +0.0245 | +0.0290 | +0.0322 | +0.18 | 3.9 passes (just before end) | WORSE |
| 8 | 36.25M | 8.15 | 1.2933 | +0.0988 | +0.1011 | +0.1194 | +0.55 | 6.9 passes | WORSE |
| 16 | 18.1M | 16.05 | 1.6648 | +0.4703 | +0.4705 | +0.5634 | +2.35 | 7.9 passes | WORSE |
| 32 | 9.06M | 31.41 | 3.6424 | +2.4479 | +2.4451 | +2.8194 | +9.63 | 7.6 passes | WORSE |

- At k = 32 the training loss reaches 0.23 nats/token while validation ends at 3.64 BPB, worse than the untrained model
  (3.46): the model memorises its 9M-token slice. Its UNSTABLE flag (P10: training-loss jump 0.149 between 30-s logs) is
  the per-pass sawtooth of memorisation, not instability (gradient norms < 1, 0.25 % clipped steps). Kept, explained.
- **R6 effective data (diagnostic):** loss law refitted on all 14 unique-data points (reproduces D24). Effective /
  seen tokens: k=2 0.87 (253M of 291M; the repeated half worth ~0.74 of new data), 4 0.69, 8 0.35, 16 0.09, 32 0.01.
  Calibration on the three unique runs 0.99-1.04x. Effective data FALLS after ~8 passes, which Muennighoff's saturating
  form cannot represent, so its fitted R* (0.6) is not meaningful here.
- **R2 (3 seeds, done 17:16):** k = 2: 1.2044 / 1.2033 / 1.1956, delta +0.0066 (not distinguishable, threshold 0.0091);
  k = 4: 1.2190 / 1.2236 / 1.2141, delta +0.0244 (WORSE). **Repetition becomes measurably worse between 2 and 4 passes**
  (both ends confirmed; no move).

### Follow-ups F2, F3 (D27; commit 3f371ee, launched 4 Nov 17:22-17:24; all clean)
| Run (36.25M unique tokens, seed 0) | Params | Passes | FWE val | vs N49 k8 s0 (1.2933) | Verdict (>0.0161) | Unique-data E02 mean | Cost of the limit | Val minimum at |
|---|---|---|---|---|---|---|---|---|
| N17 | 17.5M | 22.0 | 1.3036 | +0.0103 | inconclusive | 1.2296 | +0.074 | 19.8 passes |
| N26 | 25.6M | 13.7 | 1.2913 | -0.0019 | inconclusive | 1.2139 | +0.077 | 10.9 passes |
| N49 (reference) | 48.8M | 8.2 | 1.2933 | | | 1.1945 | +0.099 | 6.9 passes |
| N49, shifted windows (F3) | 48.8M | 8.2 | 1.2709 | -0.0223 | **BETTER** | 1.1945 | +0.077 | 6.9 passes |
| N65 | 64.6M | 6.6 | 1.2688 | -0.0245 | **BETTER** | 1.1951 | +0.074 | 5.9 passes |

- **F2 prediction missed (opposite):** with the unique data limited to 36M tokens, the smaller models were NOT better; the
  biggest (N65), which reads the slice fewest times in the hour, was best. Smaller models overfit later (their validation
  minimum comes after more passes) but still end no better.
- **F3 prediction held:** re-cutting the windows every pass removes ~0.022 of the ~0.099 damage (about a quarter).
- **Caveat (D27):** all five are single runs against a single reference run, and the reference's cost (+0.099) is the outlier
  (others +0.074-0.077). Proposed: E04-N49_rep8 seeds 1, 2, with F2/F3 reported against both seed 0 (declared) and the 3-seed
  mean (threshold 0.0132).

## Findings
1. **Repeating data hurts much earlier than Muennighoff et al. found**: already measurable at 4 passes (+0.024 with 3 seeds), large at 8 (+0.090, 3 seeds).
2. **Beyond ~7-8 passes the model memorises:** validation turns up after ~7-8 passes whatever the subset size, while the
   training loss keeps falling.
3. At 2 passes the effect (+0.0066 over 3 seeds) is below the noise threshold: 2 passes is not distinguishable from
   unique data at our resolution; 4 passes is clearly worse (+0.024).
4. **With limited unique data and a fixed hour, a smaller model does not help** (F2): N65 did best, N17/N26 no better than
   N49. The data limit costs every size about the same (~0.075 BPB).
5. **Part of the damage comes from repeating identical samples** (F3): moving the window boundaries each pass recovers ~0.022.
   Both effects (N65 better by 0.016, re-cut windows by 0.013) are at the edge of our resolution: they pass the
   declared threshold against both references but not the more conservative repeated-data threshold. More seeds were
   not run (budget; no later phase depends on them).

## Surprises, failures and what we changed
- **Prediction missed:** we expected 2 and 4 passes to be indistinguishable from unique data and 8 only slightly worse.
- The surprise prompted two follow-ups (F2, F3), declared before their runs (D26).

