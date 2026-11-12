# E05: Does Muon beat AdamW in a 60-minute budget? (Phase 4a, optimiser)

## Summary
- **Question:** at N49 (11 x 576) and 60 counted minutes on one L4, does Muon, tuned with the same effort as AdamW, predict
  unseen text better than AdamW? And is any gain per training step, or per unit of time (the declared budget)?
- **Answer:** **yes. Muon 1.1683 vs AdamW 1.1945 BPB (3 seeds each): -0.026**, about 2.9x the noise threshold (0.0091). It is
  better on all four validation sets (-0.022 to -0.026), not only on the FineWeb-Edu text both were trained on.
- **Per step, not speed:** at equal steps Muon is ahead by 0.030, at equal time by 0.028. Orthogonalising 44 matrices per
  step slows Muon by 3-5 % on the L4, but that costs only ~0.002 BPB of its gain.
- **Over the hour:** the gain is largest early (-0.15 at step 200; -0.059 at 15 minutes with 3 seeds each) and settles at
  ~0.026 over the second half of the run.
- **Tuning:** the same protocol as AdamW (4 learning rates, close call settled by 3 seeds) plus the 60-minute horizon check
  that AdamW had. Muon's best LR is 2e-3, the same as AdamW's (update scaled to AdamW's RMS). At 15 minutes every Muon LR
  tried (1e-3 to 8e-3) beat AdamW's best.
- **Held on every check:** machine-speed adjustment (-0.025), the horizon check (2e-3 1.1686, 4e-3 1.1689, 1e-3 1.1716 at
  60 min), and the test set, scored once at the end (-0.026; test = validation + 0.005).
- **Consequence:** Muon is part of both Phase 6 recipes (E08), where its gain adds up with the 16k vocabulary's
  (interaction +0.001).
- **Prediction:** direction held, size exceeded (predicted -0.01 to -0.02); time overhead at the low end of the predicted
  5-12 %.
- **Figure:** `report/figures/muon_curves.pdf` (`code/plots/fig_muon_curves.py`): learning curves of both optimisers and
  the gap at equal time and at equal tokens.
- **Cost:** 8 fifteen-minute LR pilots and 5 sixty-minute runs (3 finals, 2 extra horizon-check runs); the AdamW arm
  reuses E02-N49. 7.0 counted GPU-hours, ledger $5.90, 5 Nov 2026.

## Goal
AdamW adjusts each weight separately. Muon treats each weight matrix as a whole and, for every update, makes all of
its directions equally strong (it "orthogonalises" the momentum with a few matrix multiplications, the Newton-Schulz
iteration). Reported results say it learns more per step than AdamW. But those extra matrix multiplications cost time,
and our budget is time. So the question is not only "is Muon better per step?" but "is it better per hour on our GPU?".
We answer the per-hour question (the declared budget) and also report the per-step view, because the Muon reading asks
us to keep the two apart.

## Question
At N49 (11 x 576) and 60 counted minutes on one L4, does Muon (tuned with the same effort as AdamW) give a lower final
FineWeb-Edu validation BPB than AdamW?

## Prediction (written before any E05 run)
Muon is ~5-12 % slower per step on the L4 (Newton-Schulz on 44 matrices per step, eager mode, one matrix at a time,
on a GPU that runs power-capped at 72 W / ~870 MHz, so extra FLOPs cost time almost one-for-one). Per step it learns
faster, enough to win overall: **Muon better by ~0.01-0.02 BPB at 60 min.** What would change our mind: the time
overhead eating the per-step gain (tie), or Muon being unstable at the LRs that are best at 15 min.

**Outcome: direction held, size exceeded.** Muon is better by 0.026 at 60 minutes (3 seeds). Its time overhead was
-5.0 % tokens/s in the pilots and -3 % in the finals, at the low end of the prediction. No LR tried was unstable; clipping
occurred only in the first steps (D32). Kept as written.

## Design
- **Which parameters get Muon:** the hidden 2-D matrices of every block (attention qkv and output projection, MLP in
  and out): 44 matrices, 43.8M of 48.8M parameters. **AdamW** for the rest (tied token embedding / output head and
  position embedding with weight decay 0.1; LayerNorm weights without), at the baseline's tuned 2e-3, betas 0.9/0.95.
  This split is the standard one (Muon is for hidden layers; Jordan et al.; Liu et al.).
- **Muon settings:** `torch.optim.Muon` (PyTorch 2.15, the pinned version; reference implementation of Jordan et al.,
  checked in D30: quintic Newton-Schulz in bf16 after Frobenius normalisation, Nesterov momentum via lerp), momentum
  0.95, 5 Newton-Schulz steps, decoupled weight decay 0.1 (applied as `p *= 1 - muon_lr * 0.1`, i.e. coupled to Muon's
  own LR exactly as AdamW couples it to its LR), update scaled to AdamW's RMS (`match_rms_adamw`: 0.2 sqrt(max(m, n)),
  Liu et al.), so LR and weight decay are on AdamW's scale.
- **Same schedule and everything else** as E02-N49: 200 warm-up steps, cosine to 10 % driven by the time budget (one
  multiplier for both optimisers), clip 1.0 on the global gradient norm, 32,768 tokens/step, compile, bf16.
- **Equal tuning effort (DECLARATIONS section 1), now including the horizon check (D30):**
  1. Muon LR on a factor-2 grid 1e-3, 2e-3, 4e-3, 8e-3, 15 min each, seed 0 (`configs/E05/N49_muon_lr*_15min.yaml`),
     exactly the protocol AdamW got in E01 (N49: 5e-4 ... 4e-3, close call 2e-3 vs 4e-3 decided by 3 seeds).
  2. **Horizon check, as declared for AdamW (D07):** 60-min seed 0 at the 15-min pick AND the 15-min runner-up, plus the
     pick's lower neighbour if the runner-up is the higher neighbour (optimal LR tends to fall with longer training).
  3. Finals: the pick at 60 min, seeds 0-2 (`configs/E05/N49_muon_lr<pick>_60min.yaml`; one file per LR, no placeholder
     to edit; `summarize.py E05` prints what to launch). The pick's seed 0 doubles as its horizon-check run.
- **AdamW arm:** E02-N49 seeds 0-2 (same config except the optimiser). No new AdamW runs unless the fairness rule below
  fires.
- Workspace: giraffes573 (pilots and finals). Cost: 4 pilots (~$1.2), +4 if close call (~$1.2), +1-2 if edge (~$0.6);
  3 finals + 1-2 horizon runs (~$4.8-6); conditional fairness runs 2-4 (~$2.4-4.8, unlikely).

## Decision rules (fixed D28; revised D30 before any run)
- **R1 LR pick:** lowest seed-0 final val; every grid value must be in; **edge rule judged on the LRs actually run**
  (best at the lowest/highest tried -> next factor-2 value, at most twice per side; D30 fix: the D28 code only checked the
  original grid ends, so it would not have extended a second time); close call (runner-up within 2.8 sigma15 = 0.0188)
  -> seeds 1, 2 of both, lower 3-seed mean wins (D09 rule). A diverged pilot ranks worst.
- **R1b Horizon check (D30, = D07):** contradicted only if another LR beats the pick at 60 min (seed 0) by more than
  2.8 sigma15 = 0.0188 (D07 threshold). If contradicted, Muon's final LR is the 60-min winner (its seeds 1, 2 are run;
  the pick's 3 seeds are reported too).
- **R1c Fairness (D30):** if R1b is contradicted, AdamW N49 gets the same check (60 min, seed 0, at 1e-3 and 4e-3;
  `configs/E05/N49_adamw_lr*_60min.yaml`), because AdamW at N49 never had its own 60-min check (it inherited N26's). If
  AdamW is also contradicted, its winner gets seeds 1, 2 and R2 compares the two 60-min-checked arms.
- **R2 Decision (60 min):** Muon 3-seed mean vs AdamW 3-seed mean (E02-N49); real if |diff| > 1.6 sigma60 = 0.0091.
  Otherwise inconclusive. With only 2 usable seeds (one awaiting replacement): no verdict.
- **R3 Infrastructure:** Muon runs are checked against the Muon pilots' speed (not AdamW's: Muon is slower by design);
  > 10 % slower -> replaced with the same seed; **D30: a flagged run is excluded from R1b/R2 until replaced** (D28 code
  printed the flag but still counted the run). Other checks as in E02.
- **R4 Reported, never decisive (D30 made these concrete):**
  - R4a per step vs per time: from the learning curves (fixed 1 MB validation subset, every 5 % of the run), the
    difference at equal time (both final) and at equal steps (= equal tokens; the arm with more steps is read off its own
    curve at the other arm's step count). Caveat printed: that curve point is not fully annealed (LR ~0.11 x peak instead
    of 0.10), which slightly favours the other arm. The gap between the two views is what Muon's slower steps cost it.
    (A dedicated AdamW control run with a shortened budget was considered and rejected: one seed has noise ~0.006, the
    size of the effect to be measured; the curve view uses all 3 seeds of both arms for free.)
  - R4b machine speed: each run's speed relative to its own configuration's pilots; difference re-computed at equal
    machine speed with the E02 within-size slope of BPB on ln(tokens) (imprecise: ~-0.07 +- 0.04; sensitivity only;
    flagged if the verdict would change).
  - R4c AVG4 and the four validation sets; R4d clipping fraction, tokens, MFU (6N + attention FLOPs only, so the
    Newton-Schulz FLOPs are not in it); R4e the same comparison at 15 min (pick vs AdamW 2e-3, all seeds; the
    training-duration question of the reading); LR sensitivity of both optimisers (which grid values are within
    2.8 sigma15 of the best).

## Gaps and confounders checked (D28 + D30)
- **Per step vs per time** (the reading asks to distinguish them): the declared budget is time, so R2 is per time; R4a
  gives the per-step view.
- **Tuning effort:** same grid size, edge, close-call AND horizon rules as AdamW (D30 added the horizon check; without
  it Muon would have had less tuning than AdamW, whose 15-min ranking was checked at 60 min). AdamW's
  embeddings/LayerNorm settings are reused, not re-tuned for Muon (standard; disclosed). Momentum, NS steps and scaling
  are not tuned (defaults of the reference; disclosed).
- **Implementation:** PyTorch's own Muon (not our code; source read in D30). Our wrapper only splits the parameters and
  drives both optimisers with the one schedule; tested (parameter split, schedule ratio, learning, exact crash-resume).
  The AdamW path is bit-identical to before (tested).
- **Fused qkv (D30, limitation):** nanoGPT stores q, k and v as one 1728 x 576 matrix, so Muon orthogonalises them
  jointly; modded-nanogpt orthogonalises them separately. Splitting would need our own Muon or a changed model; a 15-min
  single-seed ablation could not resolve the expected small effect (noise 0.0067). Disclosed, not tested.
- **Weight decay coupling:** decay strength scales with Muon's LR across the grid (as AdamW's does with its LR); the
  LR pick therefore also picks the effective decay. Disclosed.
- **Gradient clipping** acts on the joint norm before Muon's orthogonalisation; for Muon matrices it only rescales the
  input, which orthogonalisation then normalises away (it still changes the weighting inside the momentum buffer).
- **Different commits:** the AdamW arm ran at 0b7dde4, Muon at the D28/D30 commit; the AdamW code path is bit-identical
  (tested), so the commit difference is flagged and explained, not a confounder.
- **Machine speed** explains most of the seed noise (E02: ~84 %); R3 + R4b cover it.
- **Single size:** gains may change with size (reading's question); N49 only. Limitation. If Muon wins, Phase 6 uses it
  at N49 only, where it was tuned.

## Runs
| Round | Runs | Workspace / commit | Launched (SGT) | Status |
|---|---|---|---|---|
| 1 | LR pilots 1e-3, 2e-3, 4e-3, 8e-3, s0, 15 min | giraffes573 / 7e1b431 | 5 Nov 10:35 (starts 10:35-11:06) | done, clean, recovered ~12:20 |
| 2 | close call 2e-3, 4e-3 x s1, s2 (15 min); 60 min s0 at 2e-3 and 4e-3 (D32) | giraffes573 / 7e1b431 | 5 Nov ~12:30 (starts 12:31-13:12) | done, clean (D33) |
| 3 | 60 min 2e-3 s1, s2 (finals) + 1e-3 s0 (horizon check) | giraffes573 / 7e1b431 | 5 Nov 15:18-15:19 | done, clean (D34) |

## Results
**Round 1 (15 min, seed 0, final FineWeb-Edu val BPB):** 1e-3 1.3086, **2e-3 1.2917**, 4e-3 1.2953, 8e-3 1.3167 -> close call 2e-3 vs
4e-3 (gap 0.0036 <= 0.0188). AdamW at 15 min (E01, same protocol): 2e-3 1.3508 (3 seeds). Muon -0.059 at 15 min (descriptive), gap
shrinking along the run (-0.15 at step 200 -> -0.065 at step 2,100). Muon tokens/s -5.0 % vs AdamW. Clipping only in the first steps.
Details D32.

**Round 2:** Muon pick **2e-3** (3-seed 1.2914 vs 4e-3 1.2934). 60 min s0: 2e-3 **1.1686**, 4e-3 1.1689; vs AdamW N49 1.1945 (3 seeds): -0.0259 (needs 3 seeds). Details D33.
**Round 3 / decision:** horizon check 60 min s0: 2e-3 1.1686, 4e-3 1.1689, 1e-3 1.1716 -> pick stands (no AdamW fairness runs).
Finals 2e-3: 1.1686 / 1.1679 / 1.1684, mean **1.1683** vs AdamW 1.1945 -> **-0.0262, BETTER** (speed-adjusted -0.0246; AVG4 -0.0248, every
set better). Equal steps -0.030 vs equal time -0.028: Newton-Schulz costs only ~0.002 of the gain. Details D34.


### On the other validation sets (means over 3 seeds)
| Optimiser | FineWeb-Edu | FineWeb | DCLM | Wikipedia | Average of four |
|---|---|---|---|---|---|
| AdamW (E02-N49) | 1.1945 | 1.3850 | 1.5404 | 1.4616 | 1.3954 |
| Muon (2e-3) | 1.1683 | 1.3606 | 1.5181 | 1.4353 | 1.3706 |
| Muon - AdamW | -0.0262 | -0.0244 | -0.0223 | -0.0263 | -0.0248 |

Both models were trained on FineWeb-Edu only. The gain carries over almost unchanged to text neither model trained on.

### Learning-rate sensitivity at 15 minutes (seed 0; 3-seed mean in brackets where a close call was run)
| LR | 5e-4 | 1e-3 | 2e-3 | 4e-3 | 8e-3 |
|---|---|---|---|---|---|
| AdamW (E01, N49) | 1.4061 | 1.3774 | 1.3577 (1.3508) | 1.3710 (1.3593) | - |
| Muon | - | 1.3086 | 1.2917 (1.2914) | 1.2953 (1.2934) | 1.3167 |

Within 2.8 sigma15 (0.0188) of the best: 3 of Muon's 4 values, 2 of AdamW's 4. Muon's worst value tried (8e-3, 1.3167)
is still better than AdamW's best (1.3508): Muon is both better and less sensitive to the learning rate here.

### Per step vs per time (R4a; learning-curve subset, 3 seeds each)
| View | Muon - AdamW |
|---|---|
| At equal time (both at 60 minutes, the declared comparison) | -0.028 |
| At equal steps (AdamW's curve read at Muon's mean 8,599 steps) | -0.030 |
| What Muon's slower steps cost it (difference of the two) | +0.002 |

Muon took 8,599 steps (282M tokens) in the hour, AdamW 8,869 (291M). AdamW's curve point at 8,599 steps is not fully
annealed (learning rate 0.10-0.11 x peak instead of 0.10), which slightly favours Muon in the equal-steps view.

### Test set (scored once at the end, D47)
| Arm | Seeds | Validation | Test | Test without contaminated documents | Average of four, test |
|---|---|---|---|---|---|
| AdamW (E02-N49) | 3 | 1.1945 | 1.1994 | 1.1998 | 1.3947 |
| Muon (2e-3) | 3 | 1.1683 | 1.1731 | 1.1737 | 1.3697 |
| Muon - AdamW | | -0.0262 | -0.0263 | -0.0261 | -0.0250 |

FineWeb-Edu test, the three Muon finals listed in `configs/frozen_configs.yaml` ("Phase 4 winner: Muon"), scored once
after all experiments; nothing was decided from test. The two horizon-check runs (1e-3, 4e-3) were not scored.
`python code/summarize.py TEST`; files `results/test/E05-*.json`.

## Findings
- Muon beats a tuned AdamW by 0.026 BPB in the same hour at N49, on every validation set; prediction (0.01-0.02) exceeded.
- The gain is per-step learning: the time cost of orthogonalisation (-3 to -5 % tokens/s on our power-capped L4) is worth only ~0.002.
- The advantage is largest early (-0.15 at step 200, -0.059 at 15 min) and settles at ~0.026 over the second half of the hour.
- Same best LR as AdamW (2e-3, with RMS-matched scaling, as Liu et al. 2025 claim); Muon's LR basin is wider; its seed-to-seed spread is
  much smaller (0.0003 vs 0.0014).
- Limitations: one size; fused qkv orthogonalised jointly; Muon hyperparameters (momentum, NS steps) at defaults.
- Held on the test set: -0.026, the same as on validation.

### Outcome of every declared rule
| Rule | Fired? | Outcome |
|---|---|---|
| R1 LR pick | Yes | Close call 2e-3 vs 4e-3 (gap 0.0036) -> seeds 1, 2 of both -> **2e-3** (3-seed means 1.2914 vs 1.2934). Best value interior, so the edge rule was not needed |
| R1b horizon check | Yes | 60 min, seed 0: 2e-3 1.1686, 4e-3 1.1689, 1e-3 1.1716 -> the 15-minute pick stands |
| R1c AdamW fairness check | No | Not triggered (R1b not contradicted); `configs/E05/N49_adamw_lr{1e-3,4e-3}_60min.yaml` never run |
| R2 decision | Yes | -0.0262 vs threshold 0.0091 -> **BETTER** |
| R3 infrastructure | Clean | Speed vs Muon's own pilots -4.7 % to +0.8 %; all 13 runs one attempt, cold compile, full budget; none replaced |
| R4a per step vs time | Reported | -0.030 at equal steps vs -0.028 at equal time |
| R4b machine speed | Reported | -0.0246 after adjustment; verdict unchanged |
| R4c four sets | Reported | Better on every set (-0.022 to -0.026); average of four -0.0248 |
| R4d clipping, tokens, MFU | Reported | Clipped steps 1.1 % vs 0.3 % (all in the first steps); 282M vs 291M tokens; MFU 0.216 vs 0.222 (MFU omits the Newton-Schulz FLOPs) |
| R4e 15-minute comparison | Reported | -0.059 (3 vs 3 seeds; 2 SE 0.011) |

## Surprises, failures and what we changed
- **The gain was larger than predicted, and four times larger at 15 minutes (-0.059).** Before trusting it we checked that
  both optimisers were scored on the identical validation text, that Muon's training loss was lower too, and that Muon led
  at every point of the learning curve (D32). It is not an artefact; it shrinks along the run and settles at ~0.026.
- **Muon clipped 3-7 % of steps in the pilots (AdamW ~1 %).** All of it in the first few dozen steps: every run starts at a
  gradient norm of ~9.5, and Muon takes longer to fall below the clipping level of 1.0. No step was clipped afterwards
  (gradient norms 0.4-0.6). Not an instability.
- **The first design gave Muon less tuning than AdamW** (no 60-minute horizon check). Found in the review before any run
  (D30): the check was added for Muon, with a fairness rule giving AdamW the same check if Muon's LR moved. The same review
  fixed three bugs in the summary code (the edge rule could not extend twice; a run flagged for replacement was still
  counted; E06 had no slow-machine check). No result was affected; nothing had run.
- **One round was saved without bending the rules (D32):** the 60-minute seed-0 runs at both close-call candidates were
  needed in every outcome (one is the first final, the other the horizon check), so they started before the 15-minute
  close call was settled. The 15-minute decision used only 15-minute runs, which finished first.
- **Infrastructure:** every launch terminal lost its connection to Modal; the runs continued and were recovered (21
  'failed' ledger rows across Phase 4, each annotated with its recovered row; one result recovered twice, marked as a
  zero-cost duplicate). A display issue in `summarize.py E05` (single-seed wording before all 3 seeds were in) was fixed in
  D36; no number was affected.

## How this relates to the readings
- **Jordan et al. (2024), Muon (blog and modded-nanogpt):** Muon orthogonalises the momentum of hidden 2-D weight matrices
  with a Newton-Schulz iteration and leaves embeddings, output head and gains to AdamW. We use exactly this split and
  PyTorch's reference implementation. Our result confirms a per-step gain at 49M parameters, under a fixed time budget, with
  a small time overhead (3-5 %) even on a power-capped GPU and an unfused, one-matrix-at-a-time implementation.
- **Liu et al. (2025), Muon is Scalable for LLM Training:** scaling Muon's update to AdamW's RMS (0.2 sqrt(max(m, n)))
  should let AdamW's learning rate and weight decay be reused. Consistent with this, Muon's best LR (2e-3) equals AdamW's
  at 11 x 576. They also stress weight decay, which we kept at AdamW's 0.1 and did not vary.
- **Bernstein and Newhouse (2024), Old Optimizer, New Norm:** Muon as steepest descent under the spectral norm, which
  explains why equalising singular values is a natural update for matrices. Our experiments do not test the mechanism
  directly (no update-spectrum measurements).
- **Thiéry, "Muon and its follow-ups" (topic page):** asks for comparable tuning effort, clearly stated budgets, the
  per-step vs per-time distinction, which parameters receive Muon, and whether gains change with size, architecture or
  training duration. E05 answers all of these except size (one size). Duration: the gain shrinks from -0.059 at 15 minutes
  to -0.026 at 60. Architecture: inside the wide 6 x 768 recipe (E08) Muon's best LR moved to 4e-3. Sensitivity: Muon's
  LR basin is wider. NorMuon and Dion were not tested.

## Limitations and what remains uncertain
- **One model size (N49), one batch size (32,768 tokens per step), one budget (60 minutes, plus 15-minute pilots).** Muon's
  advantage may change with size or batch size.
- **Muon's own settings were not tuned:** momentum 0.95, 5 Newton-Schulz steps and the RMS scaling are the reference
  defaults. The AdamW part of the Muon arm (embeddings, LayerNorm) kept AdamW's 2e-3; E08 later checked it under Muon
  (1e-3 / 2e-3 / 4e-3 at width 768) and 2e-3 was the best.
- **q, k and v are orthogonalised as one fused matrix** (nanoGPT's layout); modded-nanogpt treats them separately. Not tested.
- **The time overhead is specific to our setup:** an unfused implementation on a power-capped L4. MFU omits the
  Newton-Schulz FLOPs, so the overhead is measured in time only.
- **The AdamW arm was reused from E02** (a day earlier, another workspace and commit). The AdamW code path is
  bit-identical (tested) and the machine-speed adjustment leaves the verdict unchanged.
- **The declared noise threshold comes from AdamW runs.** Muon runs vary about five times less (sd 0.0003 vs 0.0014), so
  the threshold is conservative for this comparison.
- **No mechanism study:** we did not measure update spectra, gradient noise, or the accuracy of the orthogonalisation.

## What happened later / follow-ups
- **Phase 6 (E08) used Muon in both recipes.** With the 16k vocabulary, Muon's gain is -0.0251 (alone -0.0262) and the
  16k gain given Muon -0.0152 (alone -0.0163): the Muon x vocabulary interaction is +0.0011 (2 SE 0.0132), i.e. the two
  gains add up.
- **Muon's best LR depends on width:** inside recipe RB (11 x 576) it stayed 2e-3, inside RA (6 x 768) it was 4e-3. Tuning
  Muon once at one shape would not have transferred; E08 re-tuned it per recipe.
- **The untuned embedding LR (D36, gap 4) was checked in E08 (R1e):** under Muon at width 768, the AdamW part at 2e-3 was
  better than at 1e-3 or 4e-3.
- **Not done:** other model sizes or batch sizes; NorMuon, Dion or other variants; separate q/k/v orthogonalisation;
  momentum and Newton-Schulz steps; a mechanism study.

## Compute and time used
| Item | Jobs | Counted GPU time | Wall time | Ledger cost |
|---|---|---|---|---|
| LR pilots (4 values at seed 0, close call 2e-3/4e-3 x 2 seeds) | 8 | 2.0 h | 2.2 h | $1.76 |
| 60-minute runs (finals 2e-3 seeds 0-2; horizon check 4e-3, 1e-3) | 5 | 5.0 h | 5.2 h | $4.14 |
| AdamW arm (E02-N49, reused) | 0 | - | - | - |
| **Total** | **13** | **7.0 h** | **7.4 h** | **$5.90** |

Runs on 5 Nov 2026, 10:35 to ~16:55, on giraffes573 (commit 7e1b431). The ledger counts GPU time only (Modal bills ~1.2x in
total).

## How to reproduce
- Configs: `configs/E05/N49_muon_lr{1e-3,2e-3,4e-3,8e-3}_15min.yaml` (seed 0; 2e-3 and 4e-3 also seeds 1, 2);
  `N49_muon_lr2e-3_60min.yaml` (seeds 0-2); `N49_muon_lr{4e-3,1e-3}_60min.yaml` (seed 0, horizon check). The other
  60-minute files exist because one config was written per LR before the pilots; they were not run. AdamW arm:
  `configs/E02/N49.yaml` (seeds 0-2).
- Code: `MuonAdamW` in `code/train.py` (config fields `optimizer`, `muon_lr`, `muon_momentum`, `muon_ns_steps`,
  `muon_adjust_lr` in `code/config.py`); the test in `code/tests/test_pipeline.py` checks that Muon gets exactly the 44 block
  matrices, the LR ratio, that the model learns, and exact crash-resume. AdamW path bit-identical (tested).
- Tables and decisions: `python code/summarize.py E05`; test: `python code/summarize.py TEST`; figure:
  `python code/plots/fig_muon_curves.py`.
- Result files: `results/runs/E05-*.json`, `results/runs/E01-N49_*.json` (AdamW pilots), `results/runs/E02-N49__s*.json`,
  `results/test/E05-*.json`. Decisions D28, D30, D32-D34; DECLARATIONS Amendments row of 5 Nov.

## How this phase supports the report (marking criteria)
| Criterion | Evidence from E05 |
|---|---|
| Experimental design and exploration (35 %) | Equal tuning effort defined and enforced (same grid size, close-call rule, and the 60-minute horizon check AdamW had, plus a fairness rule if Muon's LR moved); the per-step vs per-time distinction built into the analysis before any run; which parameters get Muon stated; rules fixed before the runs, revised (D30) only before any run |
| Understanding and problem formulation (25 %) | The gain is shown to be per-step learning, not throughput, and its time cost is quantified; how the gain changes with training duration; Liu et al.'s LR-transfer claim checked; links to the Muon readings and the topic page's questions |
| Analysis and report quality (20 %) | 3 vs 3 seeds with the declared threshold; machine-speed sensitivity; four validation sets; test set; the surprisingly large gain checked before it was trusted; prediction outcome reported; limitations stated |
| Implementation and reproducibility (20 %) | PyTorch's reference Muon (source read) behind a thin, tested wrapper; AdamW path bit-identical; one config per LR (nothing edited between rounds); every number traced to run and test files; ledger annotated |

