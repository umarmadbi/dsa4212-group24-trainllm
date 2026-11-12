# E01: Learning-rate tuning for each model size

## Phase 1 summary

**Goal:** give every model size a well-tuned learning rate before the main size comparison (E02), with the same
tuning effort for every size, so that the comparison measures model size and not how well each size happened to be tuned.

**What we did and what came out of it**
1. **Rules first.** The protocol (4 learning rates x 15 min per size, edge rule, divergence rule, horizon check, noise
   threshold) was written and committed before the first pilot (D07, DECLARATIONS Amendments 2026-11-01). Every later rule
   change was recorded before the runs it affected (D09, D11, D13).
2. **48 fifteen-minute pilots + 2 sixty-minute horizon runs** on one L4 each, 1-3 Nov (N49 added later: 9 more pilots, 3 Nov).
3. **Result:** the best peak learning rate is **2e-3 or 4e-3 for every size from 7M to 65M** (N7, N11, N49, N65: 2e-3;
   N17, N26, N36: 4e-3), and within each size the two are tied within run-to-run noise. The 60-minute check at N26 gave
   the same answer as 15 minutes (4e-3 1.2058 vs 2e-3 1.2077), so no size was shifted.
4. **What changes with size is the unstable zone above the optimum:** wider models fail at lower rates (N36, width 512,
   does not learn at 8e-3; N11, width 320, still does).
5. **Run-to-run noise measured: sigma15 = 0.0067 BPB** (pooled, 24 degrees of freedom). One run is not enough to pick a
   learning rate: N11 at 8e-3 looked best with one seed and was not with three.
6. **Two infrastructure bugs found and fixed before they could damage the main experiment:** resuming crashed on the GPU,
   and Modal's container reuse let two runs skip the compile warm-up (2.9 % more training inside the same budget).

**Why it matters for the rest of the project:** this protocol became the project's standard. Every configuration tuned
later (Muon, two model shapes, three vocabularies, two final recipes) got exactly the same effort, so no later comparison
is won by better tuning. See [What happened later](#what-happened-later-how-the-phase-1-outputs-held-up).

**Cost:** ~$11.8 of Modal credit by the ledger (~$14 actual), plus ~$1.8 for the later N49 pilots
([Time and cost](#time-and-cost-of-phase-1)).

## Goal

The main experiment (Phase 2) compares 6 model sizes, each trained for 60 minutes. That comparison
is only fair if every size is trained **well**. The single setting that matters most is the
*learning rate* (how big each correction to the model is). Too big and training goes unstable;
too small and the model learns slowly and wastes the hour. Big and small models usually need
different learning rates, so using one value for all sizes would quietly favour some sizes.

So before the real comparison, we try 4 learning rates for each size in short 15-minute runs and
keep the best one per size. Every size gets exactly the same tuning effort (4 tries), as the
declarations promise. These runs are preparation: they do not count as results.

What comes out: one chosen learning rate per size, used in Phase 2.

## Question
For each of the 6 sizes, which peak learning rate (from a factor-2 grid) gives the lowest validation
bits per byte after a 15-minute run?

## Design (written before running, D07)
- **What varies:** peak learning rate, 4 values factor 2 apart:
  - N7, N11 (widths 256, 320): 1e-3, 2e-3, 4e-3, 8e-3
  - N17, N26, N36, N65 (widths 384-640): 5e-4, 1e-3, 2e-3, 4e-3
  - Why two grids: smaller/narrower models usually tolerate larger learning rates (e.g. GPT-3 and Pythia use larger LRs for smaller models). Centring each grid where the best value is likely keeps the equal effort of 4 tries meaningful. If the best value is at either edge, one more value beyond the edge is run (declared rule).
- **Held fixed:** FineWeb-Edu, 8k tokenizer, context 512, 32,768 tokens/step, AdamW (beta 0.9/0.95, weight decay 0.1, clip 1.0), warmup 200 steps then cosine to 10 % of peak, compile on, bf16, seed 0.
- **Runs:** 24 (configs/E01/), 15 min counted each, schedule compressed to 15 min.
- **Decision rule (R1):** lowest final BPB on the full FineWeb-Edu validation split, seed 0 (`python code/summarize.py E01`). The parabola vertex it prints is a diagnostic only.
- **Edge rule (R2):** best at the edge -> run the next factor-2 value beyond it; at most 2 extensions per size.
- **Diverged runs (R3):** stop automatically, ranked worst, never replaced.
- **Pilot noise (R4):** after the 24 pilots, rerun the best N26 config with seeds 1 and 2 -> sigma15.
- **Horizon check (R5):** N26 at 60 min for the best LR and the runner-up, plus the lower neighbour of the best if the runner-up is the higher one (optimal LR tends to fall with longer training, Bjorck et al. 2024). All sizes shift one grid step only if a different LR wins at 60 min by more than 2.8 x sigma15.
- **Reuse (R6):** the 60-min N26 run at the finally chosen LR is the Phase 2 N26 seed-0 run.
- Full reasoning for R1-R6: `docs/decisions.md` D07.

## Run order (plan as written before the runs; what actually happened is in the batches below)
1. **Commit the code** (training code was never committed; `launch` now refuses uncommitted code).
2. **Modal kill-and-resume test** (`configs/E00/resume_test_N7.yaml`, ~6 min, ~$0.10). Can run at the same time as step 3. Checks that a run killed on Modal restarts by itself and keeps its counted time. Needed before any 60-min run, not before the pilots.
3. **24 pilots** (`configs/E01/*.yaml`). Modal runs up to 10 at once, so they finish in about 3 rounds (~1 h total).
4. `python code/summarize.py E01` -> check flags (slow machines, slow-down within a run, edge of grid, divergence) -> edge extensions if needed.
5. **Noise seeds**: best N26 config, seeds 1 and 2.
6. **Horizon check**: 2-3 x 60-min N26 runs (configs written after step 4, committed before launch). Requires the proposed amendment in D07 to be agreed first.
7. Write the chosen LR per size into the Amendments table and the Phase 2 configs.
- **Expected cost:** 24 x ~17-19 min container time (15 min counted + compile + paused evals + loading 4.9 GB of tokens from the Volume + final eval) = ~7-7.5 GPU-hours, ~$6. Plus noise seeds ~$0.5, horizon check 2-3 GPU-hours ~$2-2.5, resume test ~$0.1. Phase 1 total ~$9 on one account (Phase 0 used ~$3 of its $30).

## Gaps and confounders checked
- **Repetition:** the largest token count in a 15-min pilot is N7 at ~380M tokens, vs 2.45B FineWeb-Edu train tokens: < 0.16 passes. No repetition.
- **Short-run bias:** the best LR for 15 min may differ from the best for 60 min (longer runs often prefer slightly lower peak LR). Covered by the horizon check.
- **Warmup is fixed in steps (200)**, so it is a different fraction of each run: ~2 % of a 15-min N7 run, ~10 % of a 15-min N65 run (N65 does fewer, slower steps). Same rule for every size, but noted: it slightly favours small models in short pilots. Recorded as a limitation; not changed mid-protocol.
- **Single seed:** pilot differences smaller than seed noise cannot be resolved. If the top two LRs for a size are within the noise level (measured later in Phase 2), either is acceptable; we take the lower BPB and note it.
- **Evaluation:** selection on validation only; the test set is never touched.
- **Grid-edge rule** applied per size; extra runs logged here.
- **Throttling (found in E00 logs, D07 item 4):** N65 slowed ~7 % within a 5-min benchmark. Now measured per run (early vs late speed + nvidia-smi telemetry). A pilot whose speed falls within the run is not replaced (it is normal L4 behaviour, same for every LR of that size), but the size's tokens/hour estimate is updated from these runs.
- **Slow machine:** a pilot > 10 % slower than the median of its size's 4 pilots is an infrastructure failure -> replaced with `--fresh`, same seed (DECLARATIONS section 6).
- **Weight decay is coupled to LR** (PyTorch AdamW multiplies decay by LR), so a higher LR also means stronger effective decay. Same for all sizes; noted as part of what "the LR" means here.
- **Paired design:** all 4 pilots of a size use seed 0 (same initial weights and data order), so differences between LRs are not inflated by init/data noise.

## Batch 1 (2026-11-01 20:32-~21:40, commit 1c690c5): 24 pilots
Launch problem first: on Windows `--configs "configs/E01/*.yaml"` failed (the Modal CLI expanded the wildcard into 24 separate arguments despite the quotes). Nothing ran. Fixed by passing a comma-separated list built in PowerShell. Cost: none.

| Size | 5e-4 | 1e-3 | 2e-3 | 4e-3 | 8e-3 | Best (declared rule) | Parabola vertex | Tokens in 15 min |
|---|---|---|---|---|---|---|---|---|
| N7 (7.0M) | - | 1.3414 | **1.3315** | 1.3354 | 1.3433 | 2e-3 (+0.0039 to 4e-3) | 2.3e-3 | 363-380M |
| N11 (11.4M) | - | 1.3213 | 1.3115 (warm) | 1.3204 | 1.3145 (warm) | **invalid: 2 runs had a free compile warm-up (D08)**, rerun | - | 239M cold / 246M warm |
| N17 (17.5M) | 1.3574 | 1.3185 | **1.3076** | 1.3079 | - | 2e-3 (+0.0002: tie with 4e-3) | 2.8e-3 | 176-178M |
| N26 (25.6M) | 1.3650 | 1.3272 | **1.3152** | 1.3172 | - | 2e-3 (+0.0020 to 4e-3) | 2.6e-3 | 126-129M |
| N36 (35.9M) | 1.3791 | 1.3405 | 1.3330 | **1.3192** | (running) | 4e-3, AT EDGE -> 8e-3 added (R2) | - | 95-98M |
| N65 (64.6M) | 1.4384 | 1.3919 | **1.3726** | 1.4203 | - | 2e-3 (+0.0193 to 1e-3) | 1.7e-3 | 57-59M |

Values: final validation BPB, full FineWeb-Edu validation split, seed 0, 15 min. None diverged; no loss spikes; no throughput flags (all within 10 % of their size's median).

### What this batch shows 
1. **The best learning rate is about 2e-3 for almost every size**, and the curve is flat near the top: for N17 and N26, 2e-3 and 4e-3 are practically tied. The parabola fits put the optimum between 1.7e-3 (N65) and 2.8e-3 (N17), a mild fall for the largest model, as expected.
2. **N36 is the odd one out** (4e-3 better than 2e-3 by 0.014, while its neighbours N26 and N65 prefer 2e-3, and N65 is clearly hurt by 4e-3). Not explained by machine speed (N36 2e-3 had ~2 % fewer tokens, worth ~0.003). Either noise is larger than we think or N36 genuinely tolerates a higher LR; the 8e-3 extension and sigma15 (R4) will tell.
3. **At 15 min the smaller models already lead** (best LR per size: N17 1.3076 < N26 1.3152 < N36 1.3192 < N11 ~1.321 (cold run) < N7 1.3315 < N65 1.3726): a 15-min version of the core trade-off, with the optimum near 17M parameters at about 10 tokens/parameter. The 60-min runs should move it larger.
4. **Two infrastructure bugs found and fixed** (D08): resume crashed on GPU; container reuse gave two N11 runs a free compile warm-up.
5. **Every L4 sits at its 72 W power cap**, at different clocks per machine -> +-2 % tokens between machines of the same size. This is real noise under a time budget (D08).

## Batch 2 (planned, after the D08 fixes are committed)
- Resume test again (`--fresh`).
- N11 lr2e-3 and lr8e-3 again (`--fresh`, replacement for the warm-compile fault).
- N36 lr8e-3 (edge extension, R2).
- N26 lr2e-3 seeds 1 and 2 (R4 -> sigma15).

## Batch 2 (2026-11-01 23:28 - 2026-11-02 ~00:35, commit e415f8c)
Queued 12-14 min for L4 capacity (Modal-wide shortage) before starting.

| Run | Why | Final BPB | Notes |
|---|---|---|---|
| E00-resume_test_N7 (--fresh) | Modal kill-and-resume test, retry after D08 fix | 1.4675 | **PASS**: killed at step 1500, auto-restarted, resumed from step 1265 / 120.0 s, ended at 240.05 s counted, 2 attempts |
| E01-N11_lr2e-3 s0 (--fresh) | replaces warm-compile run (1.3115) | 1.3152 | cold compile (first step 29.6 s), 237.7M tokens |
| E01-N11_lr8e-3 s0 (--fresh) | replaces warm-compile run (1.3145) | **1.3025** | cold compile, 248.3M tokens; now best for N11, at grid edge |
| E01-N36_lr8e-3 s0 | edge extension R2 | 2.0268 | failed to learn (loss stuck ~5.7, 74 % steps clipped) -> N36 best stays 4e-3 |
| E01-N26_lr2e-3 s1 | pilot noise R4 | 1.3215 | slow machine: 122.9M tokens |
| E01-N26_lr2e-3 s2 | pilot noise R4 | 1.3125 | 129.0M tokens |

**sigma15 = 0.0046** (N26 2e-3, 3 seeds; 2 degrees of freedom, so rough). 2.8 x sigma15 = 0.013.

### Status per size after batch 2
| Size | Best (seed 0) | Runner-up | Gap | vs 0.013 | Status |
|---|---|---|---|---|---|
| N7 | 2e-3 | 4e-3 | 0.0039 | below | close call -> 3 seeds each |
| N11 | 8e-3 (edge) | 2e-3 | 0.0127 | below | edge run 1.6e-2 + 3 seeds each |
| N17 | 2e-3 | 4e-3 | 0.0002 | below | close call -> 3 seeds each |
| N26 | 2e-3 | 4e-3 | 0.0020 | below | 4e-3 seeds 1, 2 |
| N36 | 4e-3 | 2e-3 | 0.0138 | above | **decided: 4e-3** |
| N65 | 2e-3 | 1e-3 | 0.0193 | above | **decided: 2e-3** |

## Batch 3 (planned, D09): close calls + N11 edge
15 runs: seeds 1, 2 of N7 2e-3/4e-3, N11 2e-3/8e-3, N17 2e-3/4e-3, N26 4e-3; plus N11 1.6e-2 seed 0. Pick = lower 3-seed mean of the top two LRs.

## Batch 3 + 4 (2026-11-02 12:54 - 2026-11-03 01:30): close calls; final 15-min picks
Batch 3 (commit c5dc879): 13 close-call runs + N11 1.6e-2 edge (1.3704, stalled ~5 min then partly recovered). Modal gave ~1 L4 at a time (runs started 14:06-18:56). N26 4e-3 s2 never started (laptop lost its connection at ~20:32; ledger row annotated, cost ~0).
Batch 4 (commit 43b5480, after D10 workspace tools were committed): N26 4e-3 s2 (--fresh) + N36 close call.

| Size | lr | s0 | s1 | s2 | mean | |
|---|---|---|---|---|---|---|
| N7 | 2e-3 | 1.3315 | 1.3333 | 1.3280 | **1.3309** | pick |
| N7 | 4e-3 | 1.3354 | 1.3301 | 1.3359 | 1.3338 | |
| N11 | 2e-3 | 1.3152 | 1.3120 | 1.3147 | **1.3140** | pick |
| N11 | 8e-3 | 1.3025 | 1.3155 | 1.3274 | 1.3152 | |
| N17 | 2e-3 | 1.3076 | 1.3121 | 1.3024 | 1.3074 | |
| N17 | 4e-3 | 1.3079 | 1.3020 | 1.3071 | **1.3057** | pick |
| N26 | 2e-3 | 1.3152 | 1.3215 | 1.3125 | 1.3164 | |
| N26 | 4e-3 | 1.3172 | 1.3084 | 1.3168 | **1.3141** | pick |
| N36 | 2e-3 | 1.3330 | 1.3276 | 1.3240 | 1.3282 | |
| N36 | 4e-3 | 1.3192 | 1.3169 | 1.3362 | **1.3241** | pick |
| N65 | 2e-3 | 1.3726 | - | - | - | pick (clear at seed 0) |

**sigma15 (pooled, 20 df) = 0.0062.** Picks: N7 2e-3, N11 2e-3, N17 4e-3, N26 4e-3, N36 4e-3, N65 2e-3 (D13).

### Findings 
1. **The best learning rate is 2e-3 to 4e-3 for every size from 7M to 65M**, and inside that range the difference is smaller than run-to-run noise. A small tuning budget is enough at this scale; what matters is avoiding the unstable high end.
2. **Wider models become unstable at lower learning rates:** N11 (width 320) still trains at 8e-3 (noisily), N36 (width 512) stalls at 8e-3, and N11 stalls at 1.6e-2 for its first 5 min.
3. **Near the stability limit, results get much noisier** (sd 0.0105-0.0125 vs ~0.003-0.005 at safer rates), so a single run there can look like the winner by luck: N11 8e-3 did exactly that at seed 0.
4. **Seed noise at 15 min: sigma15 = 0.0062 BPB**, including hardware differences between L4 machines (+-2 % tokens under the 72 W power cap).

## Batch 5 (launched 2026-11-03 ~01:45, overnight, --detach): horizon check
N26 at 60 min, lr 4e-3 and 2e-3 (`configs/E01/N26_lr{4e-3,2e-3}_60min.yaml`). Shift rule: all sizes one step down only if 2e-3 wins by > 0.0174.

## Batch 5 result (horizon check, 2026-11-03 03:04-~04:10)
| Run | Final BPB | Tokens | Notes |
|---|---|---|---|
| N26 4e-3, 60 min | **1.2058** | 527.7M | faster L4 (SM ~945 MHz); launch terminal lost its connection after 24 min, run continued (--detach), fetched with `recover` |
| N26 2e-3, 60 min | 1.2077 | 506.5M | SM ~875 MHz |

Learning curves (validation BPB on the 1 MB subset):
| min | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 48 | 54 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4e-3 | 1.5174 | 1.4156 | 1.3764 | 1.3479 | 1.3225 | 1.2977 | 1.2732 | 1.2477 | 1.2285 | 1.2205 |
| 2e-3 | 1.5076 | 1.4081 | 1.3599 | 1.3283 | 1.3035 | 1.2828 | 1.2612 | 1.2430 | 1.2285 | 1.2213 |

Verdict: tie (gap 0.0019 << 0.0174; partly explained by 4.2 % more tokens). **15-min picks stand; no shift (D14).**

## Conclusion

### Findings for the report
1. **One learning-rate range fits every size.** From 7M to 65M parameters the best peak LR is 2e-3 to 4e-3 at 32k tokens per step, and inside that range the differences are smaller than run-to-run noise.
2. **What changes with size is the danger zone above it.** Wider models fail at lower LRs: N36 (width 512) does not learn at 8e-3; N11 (width 320) does, but noisily, and stalls at 1.6e-2. This is the width effect behind width-aware LR rules such as muP.
3. **Single runs mislead near the edge.** N11 at 8e-3 looked best with one seed and was not with three; spread at risky LRs was 2-4 times larger than at safe ones.
4. **Run-to-run noise:** sigma15 = 0.0062 BPB (pooled, 20 degrees of freedom; 0.0067 on 24 df after the N49 pilots), part of it from hardware: L4s run at their 72 W power cap at machine-specific clocks, so equal time gives +-2-4 % different amounts of text.
5. **The 15-min choice held at 60 min**, and the 60-min curves show the typical pattern of a higher LR trailing until the final cool-down. Hence: compare configurations only on final BPB.
6. **One extra hour of a fixed model helps a lot:** N26 went from ~1.31 BPB at 15 min to ~1.21 at 60 min.
7. **Infrastructure lessons:** resume crashed on GPU (fixed), container reuse skipped compile warm-up (fixed), the Windows `*` pattern fails, laptop connection drops need `--detach` + `recover`, Modal sometimes gives ~1 L4 at a time (Phase 1 took 2.5 days instead of ~half a day).

### Phase 1 runs and cost
48 pilot runs + 2 horizon runs (the resume tests are counted in E00); see [Time and cost](#time-and-cost-of-phase-1).

## Follow-up: N49 (added by E02 decision D19, 2026-11-03)
Same protocol as the other sizes (grid 5e-4-4e-3, 15 min, seed 0; close-call rule D09).
| lr | 5e-4 | 1e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|
| N49 s0 | 1.4061 | 1.3774 | **1.3577** | 1.3710 |
Best 2e-3 (interior; parabola vertex 2.1e-3). Runner-up 4e-3 +0.0132 < 0.0174 -> seeds 1, 2 of both (D20).

N49 close call (2026-11-03 21:47-00:53, umarmadbi, commit 483249a, all clean): 2e-3 s1/s2 1.3439/1.3507, 4e-3 s1/s2
1.3545/1.3524. **3-seed means 2e-3 1.3508 (sd 0.0069) vs 4e-3 1.3593 (sd 0.0102) -> N49 pick 2e-3** (D09 rule).
Pooled sigma15 now 0.0067 on 24 df (2.8 sigma15 = 0.0188). Check: N65's seed-0 decision (gap 0.0193) still clears
the updated threshold, so no earlier pick is affected. Pattern holds: every size from 7M to 65M is best at 2e-3 or
4e-3, and the wider sizes (576-640) settle on 2e-3.

## Where the numbers come from (reproducibility)
| What | Source in the repo | Rebuild with |
|---|---|---|
| Every pilot table above, picks, close-call means, sigma15, parabola vertices, slow-machine and warm-compile flags | `results/runs/E01-*.json` (fields `final_val_bpb`, `config.lr`, `seed`, `tokens_processed`, `first_step_s`, `frac_steps_clipped`, `gpu_telemetry_summary`) | `python code/summarize.py E01` |
| Horizon check and its learning curves | `results/runs/E01-N26_lr{4e-3,2e-3}_60min__s0.json` (`final_val_bpb`, `curve`) | same |
| Configs of every run, including edge extensions and the N49 follow-up | `configs/E01/` (one file per learning rate; nothing edited between batches) | `modal run modal_app.py::launch --configs ... --seeds ...` (see `docs/REPRODUCE.md`) |
| Runs that were replaced (kept, never deleted) | `results/runs/_replaced/E01-N11_lr{2e-3,8e-3}__s0__*.json` (warm-compile fault, D08); `E01-N49_lr4e-3__s0__*.json` is an identical copy written by a second `recover` of the same run | - |
| Every GPU job, its time and cost, including failed ones | `results/compute_ledger.csv`, rows `E01-*` and unprefixed `N26_lr4e-3*` / `N49_lr*` rows (failed launch-terminal rows, annotated as recovered) | - |
| Reasoning behind every rule and pick | `docs/decisions.md` D07-D14 (main protocol and results), D19-D20 (N49) | - |
| The final picks as a declared rule | `docs/DECLARATIONS.md`, Amendments 2026-11-03 | - |

**Code version disclosure.** All E01 runs were launched from committed code (no `+dirty` row), from six commits:
`1c690c5` (batch 1), `e415f8c` (batch 2), `c5dc879` (batch 3), `43b5480` (batch 4), `9156019` (horizon check),
`50ec597` / `483249a` (N49). Between them only logging, guards, launch tooling and the two bug fixes changed (D07-D12);
the training path is the same, and every result records its commit and a hash of the code it ran.
**Ledger note:** `E01-N49_lr4e-3__s0` has two identical rows (recovered twice); count its cost once.

## Time and cost of Phase 1
Ledger estimates (GPU price only); Modal also bills CPU and memory, and actual spend was ~1.2x the ledger (D18 item 13).
All on the umarmadbi workspace.

| Batch | When (SGT) | Runs | Counted training time | Ledger cost |
|---|---|---|---|---|
| 1: 24 pilots | 1 Nov 20:32-~21:40 | 24 x 15 min | 6 h | ~$5.05 |
| 2: N11 reruns, N36 edge, N26 noise seeds | 1 Nov 23:28-2 Nov 00:35 | 5 x 15 min | 1.25 h | ~$1.06 |
| 3: close calls + N11 edge | 2 Nov 12:54-~20:10 (one L4 at a time) | 14 x 15 min | 3.5 h | ~$2.94 |
| 4: last N26 seed + N36 close call | 2 Nov 21:31-3 Nov 01:30 | 5 x 15 min | 1.25 h | ~$1.05 |
| 5: horizon check | 3 Nov 03:04-~04:10 | 2 x 60 min | 2 h | ~$1.65 |
| **Phase 1 total** | **1 Nov 20:32-3 Nov 04:10** | **50 runs** | **14 GPU-hours** | **~$11.75 ledger, ~$14 actual** |
| N49 follow-up (for E02, D19-D20) | 3 Nov 16:39-4 Nov 00:53 | 8 x 15 min | 2 h | ~$1.76 |

Elapsed time (2.5 days for ~14 GPU-hours) was set by Modal's L4 supply, not by the work: on 2 Nov Modal granted about
one L4 at a time. None of this counts toward the 60-minute per-run budget: the brief excludes tuning runs.

## Limitations
- **Coarse grid.** Factor-2 steps; the true optimum can sit between grid points (parabola vertices 1.7e-3 to 2.9e-3).
  Because 2e-3 and 4e-3 are tied within noise at every size, the cost of picking the "wrong" one is a few thousandths of
  BPB, below the noise of a single 60-minute comparison.
- **Only the peak learning rate was tuned.** Warm-up (200 steps), betas, weight decay (coupled to the LR in PyTorch's
  AdamW), batch size (32,768 tokens) and the schedule shape were fixed for every size.
- **15-minute pilots, not 60.** Compile and warm-up take a larger share of a short run (3-5 % and 2-10 %, more for the
  largest sizes). The horizon check covered this at one size (N26) only; the other sizes assume the same transfer.
- **N65 was decided on one seed** (its gap 0.0193 cleared the threshold 0.0188 by a hair once sigma15 was updated).
- **The noise threshold pools all sizes**, while risky learning rates were 2-4 times noisier than safe ones (sd
  0.0105-0.0125 vs ~0.003-0.005).
- **Tuned on FineWeb-Edu only.** Checked later for FineWeb and DCLM (E03 R6: no difference).
- **One GPU type.** Machine-to-machine speed differences (+-2-4 % tokens) are part of every number here.

## What happened later: how the Phase 1 outputs held up
| Phase 1 output | What later experiments showed | Where |
|---|---|---|
| Learning rate per size | All E02 60-minute runs trained cleanly at their picks (clipped steps 0.08-0.32 %, no divergence, no instability flag) | E02 |
| "2e-3 or 4e-3 fits everything" | Held for every configuration tuned afterwards with the same protocol: N49 2e-3; Muon 2e-3; 6 x 768 and 19 x 448 2e-3; vocabularies 4k / 16k / 32k 2e-3; final recipes RA 4e-3 and RB 2e-3 | E02, E05, E06, E07, E08 |
| Tuned on FineWeb-Edu | 4e-3 vs 2e-3 on FineWeb and DCLM differed by only 0.0017 and 0.0012 (threshold 0.0188): the pick transfers across corpora | E03 (R6) |
| The tuning protocol | Reused unchanged for every later comparison, including the horizon check for Muon. This is what makes "Muon beats AdamW" a fair claim: both were tuned with the same effort | E05, E06, E07, E08 |
| sigma15 = 0.0067 | Used as the single-seed threshold (2.8 sigma15 = 0.0188) in E03, E05 and E08. The 60-minute noise turned out about the same (sigma60 = 0.0058, E02): longer runs did not make results less noisy, because most of the noise is machine speed (84 % of N26's seed variance) | E02 (P8) |
| 15-minute runs at the picked rates | Reused as data: with E02's 60-minute runs they show the best size moving from ~17M at 15 min to ~50-70M at 60 min, and they are half of the 14 runs in the scaling-law fit | E02 (P7, D24), `code/fit_scaling_law.py`, Figure `report/figures/size_vs_bpb` |
| Horizon run E01-N26_lr4e-3_60min | Counted as E02's N26 seed 0 (declared before E02, D15) | E02 |

## Connection to the literature
- **Published learning rates for models of this size are lower** than ours: GPT-3 Small (125M) used 6e-4 and Pythia
  70M-160M used 1e-3 to 6e-4, at much larger batches and far longer training (Brown et al. 2020; Biderman et al. 2023).
  Our optimum, 2-4e-3, is 2-6 times higher. A likely reason, not tested here, is the very short horizon: Bjorck et al.
  (2024) find the best learning rate falls as training gets longer, so a 15-60-minute run prefers a higher one. Our own
  horizon check (15 -> 60 min, x4 tokens) moved it by less than the noise, consistent with a weak effect over a factor of 4.
- **The unstable zone moves down with width** (N36 fails at 8e-3, N11 does not). This is the effect that width-aware
  parametrisations such as muP are designed to remove (Yang et al. 2022), and Wortsman et al. (2023) show that such
  high-learning-rate instabilities already appear in small models. We did not try muP (follow-up below).
- **The higher learning rate trails for most of the hour and catches up during the final decay** (horizon curves above).
  This is the pattern Hägele et al. (2024) describe: a large part of the loss improvement happens in the cool-down. It is
  why configurations are compared only on final BPB, never on mid-run curves.

## Sources used in Phase 1
- Bjorck, Benhaim, Chaudhary, Wei and Song (2024/2025), *Scaling Optimal LR Across Token Horizons*, ICLR 2025, arXiv:2409.19913: horizon check design (the lower-neighbour rule, R5).
- Yang et al. (2022), *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (muP), arXiv:2203.03466.
- Wortsman et al. (2023), *Small-scale proxies for large-scale Transformer training instabilities*, arXiv:2309.14322.
- Hägele et al. (2024), *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*, NeurIPS 2024, arXiv:2405.18392.
- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3), Table 2.1; Biderman et al. (2023), *Pythia*, arXiv:2304.01373: published learning rates by model size, used to centre the grids.
- Karpathy, nanoGPT: AdamW settings (betas 0.9/0.95, weight decay 0.1, clip 1.0) kept from it.

## Surprises, failures and what we changed (consolidated)
| What happened | Why | What we changed | Where |
|---|---|---|---|
| `--configs "configs/E01/*.yaml"` launched nothing on Windows | The Modal CLI expanded the wildcard into 24 arguments despite the quotes | Comma-separated lists | Batch 1 |
| First Modal kill-and-resume test crashed on resume | Loading the checkpoint onto the GPU also moved the saved random-number state there; PyTorch only accepts it on the CPU. The CPU tests could not see this | RNG state moved to CPU before restoring; retest passed (resume from 120.0 s counted, 240.05 s total) | D08, D09, E00 |
| Two N11 pilots trained ~27 s longer than the others | Modal reused a finished container, so `torch.compile` found its kernels cached and skipped the warm-up | One run per container (`single_use_containers=True`); every result records it; summary flags a warm start; both runs rerun with `--fresh` (old results kept) | D08 |
| N36 did not learn at 8e-3 (BPB 2.03) | Too high for width 512: loss stuck at ~5.7, 74 % of steps clipped | Nothing to fix: the edge rule found the stability limit; N36 stays at 4e-3 | D09 |
| N11 at 8e-3 looked best with one seed (1.3025) and was not with three (mean 1.3152) | Near the stability limit results are 2-4 times noisier | Declared close-call rule applied to every size below the noise threshold (3 seeds each) | D09, D11 |
| Noise was larger than planned (sigma15 0.0046 on 2 df -> 0.0067 on 24 df) | First estimate from 3 runs; part of the noise is machine speed | Threshold re-computed as data came in; N36 became a close call too | D11, D13 |
| Same config and seed gave different results | GPU arithmetic is not bit-exact, and the time-based schedule maps steps to LR differently on faster or slower machines | Seeds treated as noise samples, not as exact replays; D07's "paired design" claim corrected | D09 |
| Batch 3 took ~7 hours instead of ~40 min | Modal had about one free L4 at a time | Nothing to change in code; overnight runs with `--detach`, second workspace for later phases | D11 |
| The laptop's connection dropped three times | Unstable Wi-Fi | `--detach` + `recover`; `rebuild_data` changed to hand off its job | D11, D16 |
| A job that never started was logged at ~$6 | For failed jobs the ledger used time since launch, which included hours of queueing | Failed jobs no longer get a guessed cost; row annotated | D12 |
| Training jobs asked for 32 GB of RAM | Never measured | Peak RAM now logged (9.6 GB); request lowered to 16 GB | D12, D14 |

## Follow-ups (status at the end of the project)
- **N49 learning rate:** done (2e-3, 3-seed close call; section above).
- **Does the pick transfer to other corpora?** Done in E03 (R6): yes.
- **Horizon check at other sizes:** not done (budget). The N26 result and E05's Muon horizon check (pick unchanged) suggest
  the 15-minute picks transfer, but this is untested for N7 and N65, the sizes furthest from N26.
- **Tuning anything besides the peak learning rate** (warm-up, weight decay, batch size, schedule shape): not done. Listed
  as a limitation of the whole project.
- **Width-aware learning rates (muP)** to remove the width-dependent instability: not tried.

## How this phase supports the report (marking criteria)
| Criterion | Evidence from Phase 1 |
|---|---|
| Understanding and problem formulation (25 %) | Why the size comparison needs per-size tuning; the horizon effect (Bjorck et al.), width and stability (muP, Wortsman et al.), cool-down (Hägele et al.); why our rates are higher than published ones |
| Experimental design and exploration (35 %) | Equal tuning effort, written before running; edge, divergence, close-call and horizon rules; follow-ups triggered by findings (edge extensions, close calls, N36 threshold update, N49) |
| Analysis and report quality (20 %) | Noise measured and used to call ties instead of winners; single-seed winner's curse shown (N11); limitations listed; equal time is not equal compute (machine speed) |
| Implementation and reproducibility (20 %) | Every number traced to a result file, config, commit and code hash; replaced runs kept; failures and their fixes recorded |
