# E05: Does Muon beat AdamW in a 60-minute budget? (Phase 4a, optimiser)


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

## Findings
- Muon beats a tuned AdamW by 0.026 BPB in the same hour at N49, on every validation set; prediction (0.01-0.02) exceeded.
- The gain is per-step learning: the time cost of orthogonalisation (-3 to -5 % tokens/s on our power-capped L4) is worth only ~0.002.
- The advantage is largest early (-0.15 at step 200, -0.059 at 15 min) and settles at ~0.026 over the second half of the hour.
- Same best LR as AdamW (2e-3, with RMS-matched scaling, as Liu et al. 2025 claim); Muon's LR basin is wider; its seed-to-seed spread is
  much smaller (0.0003 vs 0.0014).
- Limitations: one size; fused qkv orthogonalised jointly; Muon hyperparameters (momentum, NS steps) at defaults.

