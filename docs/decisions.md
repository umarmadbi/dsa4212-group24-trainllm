# Decisions and plan changes

Every change to the plan is recorded here with its reasoning before it takes effect.
Formal changes to budget, data, evaluation or run rules are also added to the
Amendments table in `DECLARATIONS.md`.

Template:

### D<NN> (YYYY-MM-DD): <short title>
- **Problem / trigger:** what we found (link the experiment doc or run).
- **Options considered:**
- **Gaps checked:** what could go wrong with each option, effect on fairness of earlier comparisons.
- **Decision:**
- **Affects:** which experiments/runs; whether earlier runs must be redone.

---

### D01 (2026-10-30): Initial plan
See `DECLARATIONS.md` (commit 5f8e548).

### D02 (2026-11-01): Data preparation design (code/prepare_data.py)
- **Problem / trigger:** writing the Phase 0 data script exposed details DECLARATIONS.md §3-4 leave open.
- **Decisions and the gap each one closes:**
  - *Pinned dataset revisions.* HF commit hash recorded per dataset and used for every download, so reruns read identical bytes.
  - *Eval documents = uniform random sample (bottom-k on a keyed hash, seed 1234) of the first source file*; Wikipedia uses 3 spread-out files because its files are ordered by article ID (old articles first). Eval splits do not depend on how much training data is downloaded: growing the train download later leaves val/test byte-identical (tested).
  - *Exact dedup across corpora, not only across splits.* FineWeb-Edu is a filtered subset of FineWeb, so the same document can be in FineWeb-Edu test and FineWeb train. Any train document matching any eval document of any corpus is dropped and counted in the manifest.
  - *Within-train exact duplicates are kept but counted* (declaration says dedup across splits only).
  - *`<|endoftext|>` is not a tokenizer token.* BPE has 8,191 tokens; id 8191 = EOT, appended by our code. A document containing the literal string can never create a fake boundary (tested).
  - *Per-token byte lengths saved* (`token_bytes.npy`); every document is checked so token bytes sum exactly to its UTF-8 length. This is the BPB denominator.
  - *Same sampling seed for every corpus*: a document shared by FineWeb and FineWeb-Edu gets the same key, so it tends to be in both eval sets. Harmless (counted as `eval_docs_shared_between_corpora`).
- **Gaps still open:** 13-gram near-duplicate check (before final eval); whether "first file" of each sample is representative (FineWeb sample files may be grouped by crawl dump; check the dump mix in the manifest if time).
- **Affects:** all experiments (data foundation). No runs yet.

### D03 (2026-11-01): Training-loop implementation choices (code/train.py)
- **Data order = shuffled pass over non-overlapping 513-token windows**, reshuffled each pass with (seed, pass). nanoGPT samples random windows with replacement, so "passes over the data" would be only approximate (Reading 2 guide warns about this). Our repetition experiment needs exact passes.
- **Train split loaded into RAM before the clock starts** (reading random windows from a network volume inside the loop would make speed depend on storage, not the model). Loading is setup, excluded like model construction.
- **Windows cross document boundaries** (standard nanoGPT packing; attention can see the previous document after an EOT). Same for every run; stated in the report.
- **Clock synchronises the GPU every step** (needed for the time-based schedule and exact stop). Small, equal overhead for all runs.
- **Mid-run curve uses the first 262,144 tokens (~1 MB) of FineWeb-Edu val**, every 5 % of budget; full val splits only at the end. Final model evaluated, best-curve value logged as diagnostic only.
- **`from_pretrained` deleted from model.py**: no code path can load GPT-2 weights.
- **Every launch auto-records**: `modal_app.py::launch` saves results/runs/<run_id>.json and appends a ledger row (failed runs too), marking `+dirty` if launched from uncommitted code.
- **Benchmark batch = 32,768 tokens per step** (64 x 512). Not yet the final choice: batch size is fixed in Phase 1 pilots (Amendments table).

### D04 (2026-11-01): compile on; batch 32,768 tokens/step for every size
- **Context:** E00 benchmark results.
- **compile:** S3 with compile: 112k vs 93k tokens/s (+20 %); warm-up 27 s vs 1 s. Over 60 min: ~401M vs ~336M tokens (+19 %). Warm-up across sizes 23-45 s (<1.3 % of the hour). Peak memory also lower (4.5 vs 5.4 GB). Risk checked: evaluation uses the uncompiled model so no mid-run recompilation; shapes are fixed so no recompiles during training.
- **batch:** micro-batch 64x1 vs 32x2 for S1: 420k vs 426k tokens/s (-1.5 %, noise) -> a bigger micro-batch buys no speed, so there is no hardware reason to raise the batch. Statistically, Kaplan et al.'s critical-batch estimate at our loss levels is in the hundreds of thousands of tokens, so 32k tokens/step is in the sample-efficient regime (each token used well; costs more optimizer steps, which are cheap here). Same batch for all sizes so the size comparison is not confounded by batch. LR is tuned per size at this batch (Phase 1).
- **Alternatives considered:** batch scaled with model size (confounds size with batch); larger batch for small models to raise GPU utilisation (no speed gain measured).
- **Open:** batch-size experiment possible in Phase 4 (does a larger batch + higher LR win at equal time?).

### D05 (2026-11-01): model-size grid for Phase 2 = 6 sizes, 6.9M-64.6M
- **Context:** E00 tokens/hour per size.
- **Prediction to test:** Chinchilla (~20 tokens/param; formula N_opt ~ sqrt(C/120) with C = 6ND from measured throughput) gives ~25-30M for our hour.
- **Uncertainty considered:** small-scale studies disagree on the optimal tokens/param (Kaplan: favours bigger models; Porian et al. / Besiroglu et al.: Chinchilla-style estimates shift with setup; some small-model studies favour much more data per parameter). And our compute is equal TIME, not equal FLOPs: small models use the GPU less efficiently (MFU 18 % vs 26 %), which pushes the time-optimum smaller than the FLOP-optimum. So the grid must allow the optimum anywhere from ~7M to ~65M.
- **Choice:** 6.9M (6x256), 11.4M (7x320), 17.5M (8x384), 25.6M (9x448), 35.9M (10x512), 64.6M (12x640). Roughly log-spaced (ratios 1.65, 1.53, 1.46, 1.40, 1.80), densest at the predicted optimum, enough points (>=5) to fit a curve to final loss vs log(size) as in Chinchilla's IsoFLOP method. Width/depth ratio kept ~43-53 so shape does not change much across sizes (depth vs width is tested separately at the best size).
- **Dropped:** 91.6M and 123.6M (2 and 1 tokens/param in an hour; far beyond the predicted optimum; each would cost a full LR tuning set). **Edge rule:** if 6.9M or 64.6M wins, run one more size beyond that edge (~4M or ~92M) before concluding.
- **Cost:** Phase 1 LR tuning 6 sizes x 4 LRs x 15 min = 6 GPU-h + 2 x 1 h horizon check ~ $6.5; Phase 2 6 runs + 4 extra noise seeds ~ $8. Modal Starter allows 10 concurrent GPUs per account (pricing page); E00 showed L4 availability, not the limit, staggered starts.
- **Dependency found:** Phase 1 must wait for the full data download. A 15-min pilot of the 6.9M model reads ~380M tokens; on the current 103M-token sample that is ~3.7 passes, so repetition would contaminate the LR choice.

### D06 (2026-11-01): gap review before Phase 1
Critical review of everything built so far, before launching E01. Gaps found and how each is handled:
1. **Runs launched from uncommitted code** (E00 benchmarks are `+dirty`). DECLARATIONS section 1 says every run launches from a committed git version, and the report must match the repo. E00 benchmarks are preliminary and their code is unchanged in what they measured, so they stay (disclosed). **From E01 on: commit before launching.**
2. **Kill-and-resume never tested on Modal itself** (only locally). A real preemption during a 60-min final run must resume correctly from the Volume. **Test before Phase 2** (cheap: start a short run, stop it after a checkpoint, relaunch).
3. **Relaunching a config resumes it.** A run whose directory already exists on the Volume resumes from its checkpoint (correct for crashes), but a deliberate fresh rerun (e.g. replacement for a slow machine) needs a new run name. Add a `fresh` option before Phase 2.
4. **LR ties vs noise:** with one seed, two LRs whose 15-min scores differ by less than seed noise cannot be ranked. **After E01, run the best N26 pilot with 2 more seeds** (~$0.6) to measure 15-min noise, and treat smaller differences as ties (pick by the declared rule, note it).
5. **Compile warm-up is a bigger share of 15-min pilots** (23-45 s = 3-5 %) than of 60-min runs (<1.3 %), and differs by size. Same rule for all; noted as a pilot limitation. The 60-min horizon check covers the transfer.
6. **Thermal throttling** not visible in 5-min benchmarks. Check throughput over time in the 15-min pilots (train_log has tokens vs time).
7. **Within-train duplicates** (0.77 % in FineWeb-Edu): kept and disclosed (see E00).

### D07 (2026-11-01): Phase 1 pre-launch review - code fixes and E01 rules fixed before any E01 run
- **Context:** full review of the repo at the start of Phase 1 (all docs, code, configs, ledger, E00 result files).
- **Gaps found that D06 did not cover, and what was done:**
  1. **"Committed code" was not actually checked.** Until commit 3a293c3 (~20:00 on 1 Nov) the training code was untracked, and `launch` only looked at tracked files (`--untracked-files=no`), so a run using a new, never-committed file would have been logged as clean code. (The E00 benchmarks were correctly marked +dirty because tracked files had also changed.) **Fix:** the dirty check now covers untracked files in `code/`, `configs/`, `modal_app.py`, `requirements.txt`; `launch` **refuses** to start from dirty code (override `--allow-dirty`, marked `+dirty`, for throwaway tests only); a SHA-256 of the shipped code is stored in every result and ledger row, and the config file's SHA-256 in every result. **Consequence for the workflow:** code and configs must be committed *before each launch* (docs/ stays local; results/ can still be committed at the end of the session).
  2. **Diverged runs.** A learning rate at the top of a grid can blow up (NaN). The old loop would keep training NaNs for the whole 15 min (paid), and `summarize.py` used `min()` on values that could be NaN, which can silently rank a NaN run as best. **Fix:** divergence guard (non-finite loss or gradient norm -> stop at once, `diverged: true`, no BPB, ledger status `diverged`); summaries rank diverged runs worst. A diverged run is a valid pilot result (that LR is too high), not an infrastructure failure, so it is never replaced.
  3. **Instability is only visible if it is logged.** **Fix:** gradient norm before clipping (mean and max per log interval) and the fraction of clipped steps are now logged. This shows how close an LR is to instability, which helps interpret close calls.
  4. **The L4 slows down during a run (found in E00 data, not noticed before).** Re-reading the E00 train logs: S4 (= N65, in our grid) fell from 73k to 68k tokens/s within 5 min (-7 %); S6 -3 %; small models flat. The L4 is a 72 W card, so power/thermal throttling is the likely cause. Effects: (a) E00's tokens-per-hour estimate for N65 is probably too high; (b) the >10 % slow-machine rule must not fire on normal throttling. **Fix:** a background thread samples SM clock, temperature, power and throttle reasons from nvidia-smi every 30 s (CPU side, not on the GPU path); results store speed in the 10-35 % and 65-90 % windows of each run; `summarize.py` flags runs that slow down by > 5 % within the run. E01 (15 min) and the 60-min horizon runs will show how large this is per size.
  5. **No throughput reference for N11 and N26** (they were not benchmarked in E00). **Rule:** in E01 a pilot is "slow" if > 10 % below the median of the 4 pilots of its size (the LR does not change speed). `summarize.py` flags it.
  6. **Kill-and-resume on Modal (D06 item 2) and fresh reruns (D06 item 3).** Implemented: `configs/E00/resume_test_N7.yaml` hard-kills its own process once (`os._exit`, no cleanup, like a lost machine) at ~140 s counted, after two checkpoints; Modal should restart it and it must resume with its counted time. `launch --fresh` archives an existing output directory (never deletes) before a deliberate rerun. Local tests added for both mechanisms (12 tests).
  7. **A `--fresh` rerun would have overwritten the old result JSON** (same run_id). Fix: the previous file is moved to `results/runs/_replaced/` first (README rule 3).
  8. **Run-ID format in docs/README.md** (`E01-lr3e-3-30M-s0`) differs from what the code writes (`<run_id in config>__s<seed>`, e.g. `E01-N26_lr2e-3__s0`). The code's format is the real one.
- **Rules for E01, fixed before any E01 run (pre-registration):**
  - **R1 Selection** (unchanged): lowest final BPB on the full FineWeb-Edu validation split, seed 0. A parabola through the best LR and its two neighbours (BPB vs log2 LR) is reported as a diagnostic of where the optimum lies between grid points; it never changes the choice.
  - **R2 Edge rule** (extends the declared "one more value"): if the best LR is at the edge of the grid, run the next value (factor 2) beyond it; if that is best again, extend once more (at most 2 extensions per size). All extra runs logged. Reason: the size comparison is only fair if every size's LR is genuinely near its optimum; an untuned size biases Phase 2 far more than one or two extra pilots cost (~$0.2 each). If a size is still at the edge after 2 extensions, stop and investigate (likely a bug or instability).
  - **R3 Diverged** = worst; never replaced.
  - **R4 Seed noise of the pilots** (D06 item 4): after E01, rerun the best N26 pilot with seeds 1 and 2 -> sd over 3 seeds = sigma15. Note: all LR pilots of a size share seed 0 (same initial weights, same data order), so LR-vs-LR differences are *paired* and less noisy than sigma15 suggests; sigma15 is therefore a conservative yardstick. Used only to describe how decisive each pick was, and in R5.
  - **R5 Horizon check** (makes the declared rule precise): at N26, 60-min runs of (a) the best 15-min LR and (b) the 15-min runner-up; **plus (c) the lower neighbour of the best if the runner-up is the higher neighbour.** Reason: the main known effect is that the optimal LR *decreases* as training gets longer (Bjorck et al. 2024, ICLR 2025, "Scaling Optimal LR Across Token Horizons", arXiv 2409.19913). If the runner-up happens to be the higher LR, the declared top-2 check cannot detect the expected shift. "Disagree" is defined as: a different LR wins at 60 min **by more than 2.8 x sigma15** (the declared single-seed threshold). Then every size shifts one grid step that way. A flip smaller than that is a tie: the 15-min choices stand, and this is reported. Cost: 2-3 GPU-hours.
  - **R6 Reuse:** the 60-min N26 run at the LR finally chosen for N26 has exactly the Phase 2 N26 seed-0 configuration (same code commit, data, seed, budget), so it **is** that Phase 2 run and is not repeated. Declared here, before it runs, so this is not a post-hoc choice.
- **Proposed amendment needing the group's confirmation (before the first 60-min run, i.e. before the horizon check):** DECLARATIONS says "no amendment is allowed after the first final run except to fix a bug". The horizon check is the first 60-minute run, so read literally this would freeze the design of Phases 3-5 before we have any Phase 2 results, contradicting the plan's adaptive design (follow-ups on surprising results earn marks). Proposed wording: *"The rules governing a comparison are fixed before that comparison's first run and never changed afterwards. Rules for experiments not yet started may be added later, recorded in decisions.md and the Amendments table before their runs. Bug fixes are disclosed."* This keeps the protection that matters (no changing the rules of a comparison after seeing its results) and is reported openly.
- **Affects:** all runs from E01 on. E00 results unaffected (benchmarks measured speed; the code changes add logging and guards only; the extra per-step `.item()` for the gradient norm adds one GPU sync per step, a cost of well under 1 % that applies equally to all E01+ runs, and E01 sets the throughput references).

### D08 (2026-11-01, 23:30): two infrastructure bugs found by the first E01 batch; fixes and reruns
- **Bug 1 - resume crashed on the GPU (found by the Modal kill-and-resume test, as intended).** The kill worked and Modal DID restart the job automatically on a new L4 ("Runner failed with exit code 1" -> new container). But the restarted run crashed while loading the checkpoint: `TypeError: RNG state must be a torch.ByteTensor`. Cause: the checkpoint is loaded with `map_location="cuda"`, which moves every tensor to the GPU, including the saved random-number states, which PyTorch only accepts on the CPU. The local test runs on CPU, so it could not see this. **Consequence if not caught:** every real preemption of a 60-min run would have ended in a crash instead of a resume. **Fix:** `set_rng_state` moves the states to CPU first. **Check:** rerun the resume test (`--fresh`) before any 60-min run. A manual relaunch at 23:18 ran the old code and is expected to fail the same way (ledger row kept).
- **Bug 2 - container reuse gave two pilots a free compile warm-up.** Modal reuses a container that has finished one job for the next queued job. torch.compile's kernels were then already cached, so the run skipped the ~27 s warm-up and trained ~27 s longer inside its 900 s budget. Evidence: E01-N11_lr2e-3 and N11_lr8e-3 have first step 0.11 s (others 17-33 s); identical steady speed to the other N11 runs (273k tokens/s) but +2.9 % tokens (246M vs 239M). They are exactly the two best N11 runs, which explains N11's zig-zag pattern (1e-3 1.3213, 2e-3 1.3115, 4e-3 1.3204, 8e-3 1.3145). **Fix:** `single_use_containers=True` on the training function (one run per container; parameter confirmed in Modal 1.6.3); each result now records how many runs its container served (must be 1); `summarize.py` flags a warm compile (first step < 5 s with compile on).
  - **Handling:** this is an infrastructure fault that gave those runs a larger counted budget than declared, so both are **rerun with `--fresh`, same seed** (DECLARATIONS section 6 replacement rule; the trigger is the warm first step, which is known independently of the score; we would rerun them even if they had scored worst). Old results kept in `results/runs/_replaced/`.
  - **Gap check on the other 22 pilots:** all have cold first steps (17-33 s), so they are unaffected. 60-min runs would have been affected less (27 s of 3,600 = 0.75 %) but still unequally; the fix covers them.
- **Finding - the L4 runs at its 72 W power cap all the time (telemetry).** In every E01 run the power-cap reason is active in ~97-100 % of samples, SM clock ~830-1,170 MHz (rated max 2,040). Speed within a run is flat (late/early speed 0.99-1.00 for all 24), so D07 item 4 (slow-down within a run) is NOT the issue. The real effect is **between machines**: different L4s settle at different clocks under the same power cap, so pilots of the same size processed different amounts of text in 15 min (N7: 363-380M tokens, +-2.3 %; N36: 95-98M). Under a fixed-time budget this is genuine hardware noise in the result (roughly 0.002-0.004 BPB by a rough estimate). It is the same order as the LR differences near each size's optimum, so close LR calls (N17: 2e-3 vs 4e-3 differ by 0.0002) are ties. It will automatically be part of the seed-to-seed noise measured in R4 and in the Phase 2 5-seed calibration (each seed lands on a random machine), which is the right place for it. Worth a sentence in the report: "equal time" on cloud GPUs is not exactly "equal compute".
- **R2 edge extension triggered:** N36 best at 4e-3 (top of its grid) -> `configs/E01/N36_lr8e-3.yaml`.
- **R5 horizon set determined:** N26 best 2e-3, runner-up 4e-3 (higher neighbour) -> 60-min runs at 1e-3, 2e-3, 4e-3 (`configs/E01/N26_lr*_60min.yaml`). Launch conditions: resume test passes, sigma15 measured, amendment in D07 agreed.

### D09 (2026-11-02, 00:50): batch 2 results -> resume works; LR close calls get 3 seeds (declared close-call rule)
- **Resume test PASSED** (E00-resume_test_N7, commit e415f8c): hard kill at step 1500 (~140 s counted) on attempt 1; Modal restarted the job automatically on a new L4; it resumed from the 120 s checkpoint (step 1265, counted 120.0 s, not reset); finished at 240.05 s counted, 2 attempts, final BPB 1.4675 (normal for a 4-min N7 run). The RNG fix (D08) works on the GPU.
  - **Cost of a preemption, measured:** the resumed container recompiled (first step after resume at 151 s, i.e. ~31 s of counted time for the second compile). Steps between the checkpoint and the kill are redone but their time is not counted (the clock restarts from the checkpoint's 120 s). So one preemption costs a run ~25-45 s of training (~1 % of 60 min). Declared rules count compile time and only replace runs for crashes or > 10 % slow throughput, so a preempted final run is **kept and flagged** (attempts > 1 in its result), not replaced. To be disclosed if it happens.
- **N11 reruns (cold compile, D08):** 2e-3 -> 1.3152 (warm run had 1.3115), 8e-3 -> 1.3025 (warm run had 1.3145). N11 is now best at 8e-3, the TOP of its grid -> R2 extension `configs/E01/N11_lr1.6e-2.yaml`.
- **N36 at 8e-3 failed to learn** (final 2.0268; training loss stuck ~5.5-5.9 from 2 min on, 74 % of steps clipped, gradient norm 2-4 vs 0.2 at 4e-3). Not NaN, so the divergence guard rightly did not fire; it is simply a failed LR. N36's best stays 4e-3 (now interior). Finding: N11 (width 320) trains well at 8e-3 but N36 (width 512) cannot: the largest stable LR falls with width, the effect that motivates width-dependent LR scaling (muP). Worth a sentence in the report.
- **Noise is larger than the D07 plan assumed.**
  - N26 2e-3, seeds 0/1/2: 1.3152, 1.3215, 1.3125 -> sigma15 = 0.0046 (only 2 degrees of freedom: very uncertain). Seed 1 ran on a slow machine (122.9M tokens vs 129.0M for seed 2) and was worst, so machine speed is part of the noise.
  - Same config AND same seed (N11 8e-3) run twice: 1.3145 vs 1.3025, a 0.012 gap (the first had the warm-compile fault, so not a clean pair, but it trained longer and still lost). Same seed does not mean same run: GPU arithmetic is not bitwise deterministic, the time-based schedule maps steps to LR differently on machines of different speed, and the trajectories then drift apart.
  - **Correction to D07 R4:** D07 claimed that sharing seed 0 across LRs makes LR comparisons "paired" and sigma15 therefore conservative. This evidence says the pairing removes little of the noise. sigma15 is a fair, not conservative, yardstick.
- **Decision - apply the declared close-call rule to the LR pilots.** DECLARATIONS section 6: "a single-seed difference below 2.8 sigma triggers the close-call reruns" (3 seeds). With sigma15 = 0.0046, 2.8 sigma15 = 0.013. Sizes whose best and runner-up LR differ by less: **N7** (2e-3 vs 4e-3, 0.0039), **N11** (8e-3 vs 2e-3, 0.0127), **N17** (2e-3 vs 4e-3, 0.0002), **N26** (2e-3 vs 4e-3, 0.0020). N36 (0.0138) and N65 (0.0193) clear the threshold: decided (4e-3 and 2e-3).
  - **Runs:** seeds 1 and 2 of both top-2 LRs for N7, N11, N17; seeds 1, 2 of N26 4e-3 (2e-3 already has them). 14 runs + the N11 1.6e-2 edge run = 15 x 15 min (~$3.2).
  - **Selection rule for these sizes (fixed now, before the runs):** the lower mean final BPB over seeds 0-2 of the top-2 LRs. If the N11 1.6e-2 edge run beats 8e-3 at seed 0, its 2 extra seeds are run and it joins the comparison in place of the runner-up.
  - **Why spend this:** a tied LR pick costs little by itself, but at 60 min neighbouring sizes may differ by only ~0.005-0.01, so an LR picked on noise could flip the size ranking, the project's main result. It also gives a pooled sigma15 on ~16 degrees of freedom instead of 2, which the horizon check (R5 threshold) and Phase 2 planning both need. Gap considered: this gives some sizes more pilot runs than others ("equal tuning effort"). Accepted: the extra runs only re-measure LRs already in every size's equal grid, they don't search new values, and the rule that triggers them is declared and applies to every size the same way.
- **Horizon check postponed** until the N26 close call (2e-3 vs 4e-3) is resolved, since the R5 set depends on which is best: {2e-3, 4e-3, 1e-3} if 2e-3 stays best, {4e-3, 2e-3} if 4e-3 wins. The three 60-min configs already written cover both cases. Still also needs the D07 amendment agreed.

### D10 (2026-11-02, 15:45): using a second Modal workspace (a friend's credits) for Phase 2
- **What this is for, in plain words:** our own Modal credits will not cover Phase 2, so runs will also be billed to a friend's workspace. Modal keeps data per workspace, so that workspace needs its own copy of the ~15 GB token files, and we must prove the copy is byte-identical to ours before any run there counts (DECLARATIONS, "Running across several Modal accounts").
- **Gap 1 found - the rebuild would NOT have used the same dataset versions.** `prepare_data.py` called `resolve_revision()`, which asks Hugging Face for the dataset's *current* main commit. The manifest records the commit we actually used, but a rerun never read it. If any of the four datasets has been updated since 1 Nov, the friend's workspace would silently get different train/val/test text (different documents in the eval splits, different tokenizer), and every comparison across workspaces would be confounded. **Fix:** `--pin_revisions` (JSON {corpus: commit}); when given, those exact commits are downloaded.
- **Gap 2 found - no command to check the hashes** (data/README said "still to be written"). **Fix:** `modal run modal_app.py::verify_data` hashes on the Volume every file training reads (12 token files + tokenizer.json + token_bytes.npy) and compares them with `data/manifest.json`; it fails loudly on any MISSING/MISMATCH.
- **New one-step command:** `modal run modal_app.py::rebuild_data` reads the size settings and the four dataset commits from `data/manifest.json`, rebuilds, then runs `verify_data`. So nobody has to remember `--train-mb 9500`.
- **Checks done:** stub test confirms pinned commits are used (resolve_revision never called) and the unpinned path still resolves; all four corpora have a recorded revision. Not yet run on Modal: first run `verify_data` on our own workspace (must PASS: tests the check itself), then `rebuild_data` on the friend's.
- **How the friend's workspace is used:** he invites umarmadbi@gmail.com as a member (Modal Starter: up to 3 seats; inviting needs a payment method on his account). We add a second CLI login profile for his workspace and launch from our repo, so the code, git commit and ledger handling are unchanged; the ledger's `modal_workspace` column records which workspace paid.
- **Consequences to keep in mind:** (1) the runs Volume is per workspace too: a preempted run resumes only within the workspace it started in (Modal's own retry does this automatically), and checkpoints from our workspace are not visible there; results still come back to `results/` on this computer. (2) The 10-GPU limit is per workspace, so two workspaces allow up to 20 runs at once (subject to Modal's L4 supply). (3) Switching profile is global; use `$env:MODAL_PROFILE` per terminal and check `modal profile current` before each launch so runs are billed where intended.
- **Effect on runs still in progress (E01 close-call pilots, N26 60-min horizon runs):** committing D10 changes the git commit and `code_sha256` that later runs record, so `summarize.py` / the DECLARATIONS commit rule will flag runs launched after it as differing from c5dc879. The diff touches only data preparation (`prepare_data.py` split stage, the `prepare_data` function's arguments) and three new functions (`hash_data_files`, `verify_data`, `rebuild_data`); `train()`, `model.py`, `train.py`, `bpb.py`, configs and the image are unchanged, so training is identical. Flagged runs are therefore kept, with this entry as the explanation.

### D11 (2026-11-02, 20:15; numbered D10 at first, renumbered because a D10 written in parallel for the second workspace came first): batch 3 results; updated noise makes N36 a close call too
- **Batch 3 (13 of 14 close-call runs + N11 1.6e-2 edge run, commit c5dc879, all clean, cold compile).** Modal gave us ~1 L4 at a time (runs started 14:06 ... 18:56), so the batch took ~6 h instead of ~40 min. Queue time is not billed or counted.
- **N11 1.6e-2 = 1.3704** (stalled at loss ~5-5.5 for the first ~5 min, recovered only as the LR decayed) -> 8e-3 is interior; edge rule done.
- **Close-call picks (lower 3-seed mean, D09 rule):**
  - N7: 2e-3 1.3309 vs 4e-3 1.3338 -> **2e-3**
  - N11: 2e-3 1.3140 (sd 0.0017) vs 8e-3 1.3152 (sd 0.0125) -> **2e-3**. The seed-0 "win" of 8e-3 (1.3025) was luck: its seeds 1-2 gave 1.3155 and 1.3274. Finding: near the stability limit, results are much noisier (sd 0.0125 vs 0.0017 at 2e-3), so a single run at a high LR can look best by chance.
  - N17: 2e-3 1.3074 vs 4e-3 1.3057 -> **4e-3** (difference 0.0017, well within noise: a genuine tie; the rule decides)
  - N26: 2e-3 1.3164 (n=3) vs 4e-3 1.3128 (n=2, seed 2 pending)
- **Pooled sigma15 = 0.0058 on 15 degrees of freedom** (replaces 0.0046 on 2 df). 2.8 sigma15 = 0.0162.
- **Consequence (gap check): N36 is now below the threshold.** D09 treated N36 as decided because its gap (0.0138) exceeded 2.8 x 0.0046 = 0.013. With the better noise estimate the threshold is 0.0162, so by the same declared rule N36 (4e-3 vs 2e-3) is a close call -> seeds 1, 2 of N36 2e-3 and 4e-3 (4 runs, ~$0.85). N65 (0.0193) still clears it: **decided, 2e-3**. Rule applied: the threshold uses the best available noise estimate at the time of the decision; a size already settled by 3-seed means is not reopened.
- **Running unattended:** no one could stay at the laptop, so the last N26 run and the 4 N36 runs are launched with `modal run --detach` (the app keeps running when the laptop disconnects); results are fetched afterwards with `modal_app.py::recover`, which writes the same result files and ledger rows.

### D13 (2026-11-03, 01:40): final 15-min LR picks; horizon check launched overnight (D12 = memory/cost logging fixes, written in parallel)
- **Last close-call runs (commit 43b5480, clean, cold compile, 1 run per container):** N26 4e-3 s2 1.3168; N36 2e-3 s1/s2 1.3276/1.3240; N36 4e-3 s1/s2 1.3169/1.3362.
- **Pooled sigma15 = 0.0062 on 20 degrees of freedom**; 2.8 sigma15 = 0.0174. N65's gap (0.0193) still clears it.
- **15-min learning-rate picks (D09 rule: lower 3-seed mean of the top two; N65 by single-seed rule):**

| Size | Pick | 3-seed means (pick vs other) | Note |
|---|---|---|---|
| N7 | 2e-3 | 1.3309 vs 1.3338 (4e-3) | |
| N11 | 2e-3 | 1.3140 vs 1.3152 (8e-3) | 8e-3 sd 0.0125: unstable regime |
| N17 | 4e-3 | 1.3057 vs 1.3074 (2e-3) | tie within noise |
| N26 | 4e-3 | 1.3141 vs 1.3164 (2e-3) | tie within noise |
| N36 | 4e-3 | 1.3241 vs 1.3282 (2e-3) | 4e-3 sd 0.0105 (1.3169-1.3362): noisy, near N36's stability limit (8e-3 failed) |
| N65 | 2e-3 | single seed, 1.3726 vs 1.3919 (1e-3) | decided at seed 0 |

  Every pick is 2e-3 or 4e-3, and every within-size difference between the two is below 2.8 sigma15: across 7M-65M parameters the optimum is flat between 2e-3 and 4e-3 at our batch size. The picks matter less than feared for Phase 2 fairness.
- **Horizon check set (R5):** N26 pick 4e-3, runner-up 2e-3 is the LOWER neighbour, so the set is {4e-3, 2e-3} (no 1e-3 run; `N26_lr1e-3_60min.yaml` stays unused). Rule: shift all sizes one step down only if 2e-3 beats 4e-3 at 60 min by more than 0.0174; otherwise the picks above stand.
- **Status of the horizon runs: preliminary (Phase 1) work.** DECLARATIONS section 1 lists the horizon check inside the LR tuning protocol, so these are pilots, not final runs, and launching them does not trigger "no amendments after the first final run". R6 (reusing the run at the chosen LR as Phase 2's N26 seed-0 run) is **deferred**: it is decided when Phase 2 starts, together with the D07 amendment the group still has to agree. If the group does not agree, Phase 2 simply runs N26 again (~$0.85).
- **Launched overnight with `--detach`** (computer left on); results fetched in the morning with `recover`. Peak host RAM is now logged (D12), so these runs also tell us whether the 32 GB request can be lowered.

### D14 (2026-11-03, 09:50): horizon check result -> 15-min picks stand; Phase 1 closed; memory request 32 -> 16 GB
- **Horizon check (N26, 60 min, seed 0, commit 9156019, clean, cold compile, 1 attempt each):** 4e-3 **1.2058** (527.7M tokens), 2e-3 **1.2077** (506.5M tokens). The lower LR did not win, so by R5 (shift only if 2e-3 wins by > 2.8 sigma15 = 0.0174) **all 15-min picks stand**.
- **Gap check - is the 4e-3 "win" real? No, it is a tie, and that is fine.** The 4e-3 run landed on a faster L4 (SM clock ~945 vs ~875 MHz) and read 4.2 % more text. At these slopes that is worth a few thousandths of BPB, about the size of the 0.0019 gap. So at equal tokens the two are indistinguishable, perhaps with 2e-3 marginally ahead: a hint of the expected horizon effect (best LR falling with longer training, Bjorck et al. 2024), but far below noise. Either LR would be fair; the declared rule keeps the 15-min pick.
- **Learning-curve shape (report-worthy):** 4e-3 was behind 2e-3 for most of the hour (24 min: 1.3479 vs 1.3283; 48 min: 1.2477 vs 1.2430) and caught up only during the final decay (54 min: equal; 60 min: ahead). The classic pattern: a higher LR makes slower progress at its peak and gains during the cool-down. Practical consequence for Phase 2: **mid-run curves must not be used to compare configurations, only final BPB** (which DECLARATIONS already requires).
- **Final peak LR per size (written into the DECLARATIONS Amendments table):** N7 2e-3, N11 2e-3, N17 4e-3, N26 4e-3, N36 4e-3, N65 2e-3.
- **Gap noted for Phase 2 - the picks are not all "the optimum".** Between 2e-3 and 4e-3 every size is within noise, so the picks partly reflect chance. Effect on the size comparison: an LR within noise of the best costs at most a few thousandths of BPB, below the noise of a single 60-min comparison. Stated as a limitation; not worth more tuning runs.
- **Memory request lowered from 32 GB to 16 GB** for training jobs: measured peak host RAM 9.6 GB (both 60-min runs, D12 logging). Purely a scheduling setting (same GPU, same code path, same data); recorded in case it changes how fast Modal finds an L4. Phase 2 runs keep logging peak RAM, so an out-of-memory risk would be visible. prepare_data keeps 32 GB (it tokenizes in memory).
- **R6 (reuse E01-N26_lr4e-3_60min as Phase 2's N26 seed-0 run):** the chosen LR is 4e-3, so this is the run that would be reused. Still deferred until the group agrees the D07 amendment; if reused, it is identical in config, code path and data to a Phase 2 N26 run (it ran from commit 9156019, which differs from the Phase 2 commit only in logging/launch code).
- **Phase 1 cost:** about $14 of the umarmadbi workspace (credits $30 -> ~$12.9 after Phase 0 + Phase 1; check the Modal usage page for the exact figure).

### D15 (2026-11-03, 10:00): amendment policy adopted (the D07 proposal); 60-min N26 4e-3 run reused in Phase 2
- **Context:** we checked what the professor's documents require. The topic page requires the per-run compute budget (hardware + amount of computation), the initialisation and the evaluation objective to be declared before experiments; preliminary/tuning runs are excluded from the budget; validation for tuning, separate test set. The assignment guidelines contain no freeze rule and reward "investigate unexpected results ... unsuccessful attempts" and "follow-up on findings" (35 % criterion). The class notes say "very open ended". The freeze sentence was our own, from 30 Oct.
- **Decision (agreed; to be shared with the group):** adopt the D07 wording as a DECLARATIONS amendment: budget and test-set lock never change; a comparison's rules are fixed before its first run; new experiments may get new rules if recorded before their runs; bug fixes disclosed.
- **R6 adopted:** E01-N26_lr4e-3_60min__s0 (1.2058) counts as Phase 2's N26 seed-0 run. Gap considered: was it selected on its own outcome? No. 4e-3 was chosen from the 15-min 3-seed means before the 60-min runs; its 60-min score could only have changed that if 2e-3 had won by > 0.0174. Remaining difference from a fresh Phase 2 run: launched from commit 9156019 (Phase 2 will run from a later commit whose changes are logging, launch and memory-request only, not the training path). Disclosed in the report.

### D16 (2026-11-03, 12:20): data rebuild on joshuahanrc was cancelled by a dropped connection -> rebuild_data now hands the build off (spawn)
- **What happened:** `modal run --detach modal_app.py::rebuild_data` (10:52) started correctly on the joshuahanrc workspace and used the four pinned dataset revisions. The laptop then lost its internet connection (`getaddrinfo failed`, 3rd time in 2 days) and Modal shows the `prepare_data` call as **Cancelled**. `verify_data` on joshuahanrc: all 13 files MISSING (nothing is saved until the build's final Volume commit). The hash check therefore worked as intended: no run can use that workspace yet.
- **Why --detach did not save it:** `rebuild_data` waited on the build with `.remote()`. The E01 60-min run survived the same kind of drop because `launch` hands jobs off with `.spawn()` and only waits for results. Inference from these two cases, not from Modal documentation.
- **Fix:** `rebuild_data` now uses `.spawn()` and returns immediately; it must be run with `--detach`, and `verify_data` is run separately about an hour later. Training code unaffected.
- **Credits:** joshuahanrc spent $0.04 so far.
- **Default profile:** `umarmadbi` was set as the default again (`modal profile activate`), so a terminal without `$env:MODAL_PROFILE` bills our own workspace.

### D17 (2026-11-03, 12:45): Phase 2 (E02) design and decision rules, fixed before any E02 run
- Full design: `docs/experiments/E02_model_size.md`. Summary: 6 sizes x 60 min at their E01 LRs (seed 0; N26 seed 0 reused per D15), N26 seeds 1-4 for sigma60 (split across both workspaces so sigma60 includes workspace differences); 9 new runs.
- Rules: P1 lowest final val BPB at seed 0, real only if > 2.8 sigma60, else 3 seeds each and 1.6 sigma60, else "optimal range"; P2 edge sizes N4 / N92 (neighbour's LR, untuned, disclosed); P3 parabola in log10(params) as a diagnostic of the optimum (tokens/param vs Chinchilla's 20); P4 report FLOPs because equal time != equal FLOPs; P5 final BPB only for decisions; P6 depth vs width designed after P1.
- Gap considered and accepted: eval only on FineWeb-Edu validation (same as E01, so the reused N26 run is comparable); other corpora are scored at the final test evaluation.
- Joshua (joshuahanrc) is a group member, so using his credits matches DECLARATIONS "group members' accounts"; no amendment needed.
- `summarize.py E02`: per-size table (params, tokens, tokens/param, FLOPs, seed-0 and mean BPB), sigma60 and thresholds, edge flag, parabola optimum; counts the reused E01 run as N26 seed 0.

### D18 (2026-11-03, 14:30): pre-launch review of Phase 2 -> E02 rules tightened, summary fixed (before any new E02 run)
- **Context:** full review of the plan, D01-D17, code, configs and results before launching E02 (looking for gaps first).
- **Gaps found and what was changed:**
  1. **`summarize.py E02` did not apply the declared rule.** It ranked sizes by the mean over all seeds, so N26's 5-seed mean was compared with other sizes' single seeds, while P1 says seed 0. **Fix:** P1 implemented step by step (seed 0 -> close-call set -> 3-seed means of seeds 0-2).
  2. **No infrastructure checks ran for E02.** Warm compile, reused container, slow machine, preemption, dirty code were only checked for E01, so the replacement rule could not be applied to final runs. **Fix:** all checks run per E02 run; slow machine = steady speed > 10 % below that size's E01 pilot median (steady, not average, because compile is ~3 % of a 15-min run and ~1 % of a 60-min run). Commit and code hash printed per run (the E02 doc already claimed this; it was not implemented).
  3. **Close call compared only the best and the runner-up.** With a flat optimum (15-min pilots: N11, N17, N26 within 0.009) the third size can be within noise as well, and choosing the runner-up from one seed is the single-seed luck E01 exposed (N11 8e-3). **Change:** every size within 2.8 sigma60 of the best (at most 3) gets seeds 1, 2.
  4. **sigma60 rests on 4 df** (95 % range ~0.6-2.9x). **Change:** the 3-seed decision uses sigma pooled over every size with >= 2 seeds (same rule as D11 in E01). The seed-0 threshold only picks contenders.
  5. **Noise is not the same for every size.** N36's pick (4e-3) had 15-min seed sd 0.0105 vs ~0.003 at safe LRs, and N36 is next to the predicted optimum. **Change:** P10 instability flag (> 5 % clipped steps or a loss jump > 0.1 between log intervals; healthy N26: 0.16 %, 0.013); a flagged size gets seeds 1, 2. Outcome-independent of the ranking, declared now.
  6. **Tokens/param depends on the parameter convention.** Embeddings are 30 % of N7 and 8 % of N65; Chinchilla counts all parameters, Kaplan non-embedding (Porian et al. 2024 attribute part of their disagreement to this). **Change:** both reported; total is the headline.
  7. **No pre-registered prediction.** Added: 15-min optimum ~17M; x4 compute -> ~34M (Chinchilla, a ~0.5) or ~46M (Kaplan, a ~0.73), pulled smaller by low small-model MFU -> predicted 25-35M.
  8. **The grid's widest gap (N36 -> N65, x1.8) is where the prediction partly lies.** **Change:** P2b conditional refinement size N49 (11 x 576, 48.8M), tuned with the declared 4 x 15-min protocol, only if the P3 optimum falls between N36 and N65. Configs written now (`configs/E01/N49_lr*.yaml`, `configs/E02/N49.yaml`, LR placeholder 2e-3).
  9. **Free analysis not planned:** the E01 15-min runs at the picked LR are complete runs at 1/4 of the compute, so the shift of the optimum from 15 to 60 min estimates the allocation exponent (P7). Also P8: share of N26 seed variance explained by machine speed.
  10. **Baseline size for Phases 3-6 was undefined if P1 ties.** **P9 (fixed now):** the P1 winner if real; in an optimal range containing N26, N26 (its 5 seeds double as the Phase 3 FineWeb-Edu arm and the Phase 6 baseline arm); else the lowest mean.
  11. **E02 scored only FineWeb-Edu validation.** Phase 3 needs the baseline on all four sets. **Change:** E02 configs set `eval_corpora: [fineweb_edu, fineweb, dclm, wikipedia]` (headline stays FineWeb-Edu; scored after training, outside the budget; training identical). N26 seed 0 is rescored from its saved model later.
  12. **`recover` wrote this terminal's code hash, member and workspace into the ledger row**, not the run's. **Fix:** taken from the result's own metadata. Result JSONs were always correct.
  13. **Cost was underestimated.** The ledger estimate counts the GPU only; Phase 0-1 actual spend was ~1.2x it. A 60-min run costs ~$1.0, Phase 2 ~$9 + $2-6 for close calls. Remaining after Phase 2: ~$28-30 over two workspaces for Phases 3-6 (~20-25 runs): tight. **Recommendation:** onboard the other two members' workspaces now (rebuild_data ~1 h, ~$0.6 each), which also adds parallel L4s.
  14. **E02's rules had no DECLARATIONS Amendments row**, which the D15 policy requires before a comparison's first run. **Fix:** row added (D17 + D18 rules).
  15. **Workspace split rebalanced** (umarmadbi 4 runs, joshuahanrc 5) to cut wall time (~1 L4 at a time per workspace so far).
- **Considered and kept:** the 6-size grid, the 60-min budget, the LR picks (re-tuning N36 at 2e-3 would change E01's declared selection after the fact), the per-step GPU syncs (changing them would break comparability with E01 and the reused N26 run; stated as a limitation and a Phase 4 candidate).
- **Rule status:** E02 rules were fixed in D17 after the reused N26 seed-0 run but before any new E02 run; D18 changes them before any new E02 run too, and none of the changes depends on an E02 outcome (only N26 seed 0 exists, and every rule treats it identically to a fresh N26 run). Allowed by the amendment policy (D15).
- **Affects:** E02 only. No result changes.

### D19 (2026-11-03, 16:45): E02 seed-0 results; parabola bug fixed; N36/N65 close call; N49 follow-up declared before the close-call seeds land
- **Results (9 new runs, commit 90f6087, all clean: 1 attempt, cold compile, 1 run per container, speed -5.8 % to +6.0 % vs E01 pilots, no instability, no failed ledger rows despite both launch terminals losing the connection):** final FineWeb-Edu val BPB at seed 0: N7 1.2824, N11 1.2426, N17 1.2296, N26 1.2058 (5 seeds: 1.2058, 1.2193, 1.2164, 1.2127, 1.2074; mean 1.2123), **N36 1.1888, N65 1.1899**. Same order on FineWeb, DCLM and Wikipedia validation.
- **sigma60 = 0.0058** (N26, 4 df), about the same as sigma15 (0.0062), not smaller as assumed. **P8: 84 % of the N26 seed variance follows machine speed** (BPB vs ln tokens slope -0.116: +4 % tokens ~ -0.005 BPB). Seeds 1 and 2 landed on slow L4s (810-848 MHz), seed 4 on a fast one (1,058 MHz). Equal time on cloud L4s is clearly not equal compute; this is the main noise source.
- **P1 step 1-2 (declared rule):** best N36; N65 +0.0011 is within 2.8 sigma60 (0.0161); N26 +0.0170 is just outside (and its 5-seed mean is 0.0235 behind). Close-call set {N36, N65} -> seeds 1, 2 of both (4 runs).
- **Bug found: the P3 parabola fit was wrong.** `_quad_fit` (written in D17, kept in D18) ordered the right-hand side of the normal equations (1, x, x^2) instead of (x^2, x, 1). It printed an optimum of 20.7M for data whose two best sizes are 36M and 65M. My D18 synthetic test only checked that the output looked plausible, not that it matched an independent fit. **Fix:** corrected and checked against numpy.polyfit on random data (identical). No decision had used it yet.
- **Corrected P3 (declared, all sizes):** optimum ~103M, beyond the largest size, leave-one-size-out range 71M to >10,000M: a parabola over a whole decade is dominated by the small sizes and is not informative here. Reported as such. **Post-hoc local fit** (N26, N36, N65; labelled post hoc, diagnostic only): ~48M, ~7 tokens/param.
- **Error in the pre-registered reasoning (kept as written, corrected here):** D05 and D18 said low MFU of small models "pulls the time-optimum smaller". The direction is wrong. Under equal time, bigger models get MORE FLOPs (MFU 0.18 for N7 vs 0.25 for N65: 77 vs 106 PFLOPs), so the equal-time curve is tilted in favour of bigger models and the time-optimum sits LARGER than an equal-FLOP optimum. The prediction (25-35M) was therefore biased low and missed: the best sizes are N36-N65 (3.8-11.5 tokens/param). Kaplan's exponent gave ~46M, close to the post-hoc local fit; to be assessed after the close-call seeds.
- **Follow-up declared now, before the N36/N65 seeds 1-2 are known (amendment policy item 3):** the declared P2b trigger ("P3 optimum between N36 and N65") depended on the buggy fit, and the corrected whole-range fit cannot locate the optimum. The seed-0 data already place the optimum at >= 36M, and the grid has its widest gap (x1.8) exactly there. **Decision: run N49 (11 x 576, 48.8M) regardless of the close-call outcome**: 4 LR pilots x 15 min (declared protocol, `configs/E01/N49_lr*.yaml`), then `configs/E02/N49.yaml` seed 0 at the pilot pick; it joins P1 like any size (seeds 1, 2 if within 2.8 sigma60 of the best seed-0 size, final call on 3-seed means with pooled sigma). P2 still applies: if N65 has the lowest 3-seed mean and beats every contender by > 1.6 pooled sigma, N92 is run as well. The pilots are launched now, in parallel with the close-call seeds, to save a round (~1.5 h). Cost ~$1.1 pilots + ~$1.1 seed 0 (+ ~$2.2 if it needs seeds 1, 2).
- **Why this is not post-hoc selection:** N49 adds a measurement where the data say the optimum lies; it does not change how any existing comparison is judged, and its rules are fixed before its runs and before the deciding seeds. Disclosed as a follow-up prompted by seed-0 results.
- **Budget after these runs (estimate, check the usage pages):** joshuahanrc ~$19, umarmadbi ~$4. Phases 3-6 then need ~15-20 runs: onboarding the other members' workspaces is now necessary, not optional. P6 (depth vs width) is the first thing to drop if credits run short.

### D20 (2026-11-03, 20:50): N36/N65 close call = tie (optimal range); N49 LR pilots are a close call too
- **Close-call seeds (commit 50ec597, joshuahanrc, all clean):** N36 s1 1.1963, s2 1.2026; N65 s1 1.1923, s2 1.2030. 3-seed means **N36 1.1959, N65 1.1951** (gap 0.0008). Pooled sigma60 = 0.0064 on 8 df; 3-seed threshold 1.6 x 0.0064 = 0.0102. **P1: optimal range {N36, N65}** (provisional: N49 still has to join).
- **Seed-0 winner's curse, seen directly:** the seed-0 runs of the two winners were their best seeds (N36 1.1888, N65 1.1899), both on faster L4s; their later seeds were 0.004-0.014 worse. Selecting the top sizes on one seed overstates them, which is why the rule re-measures them with 3 seeds.
- **Machine speed again:** seed 2 of both sizes ran on slow L4s (825-840 MHz) and is each size's worst run. Pooled within-size slope over N26/N36/N65 (11 runs): -0.149 BPB per unit ln(tokens). Diagnostic only.
- **P2 (edge):** N65 has the lowest mean but does not win outright, so by the rule declared in D19 (before these seeds) **N92 is not run**. `summarize.py` printed "run N92" because it still implemented the D18 wording; corrected to D19's rule. Reported as a limitation: the plateau may continue above 65M.
- **P9 (provisional):** baseline for Phases 3-6 = lowest mean in the range = N65 (0.0008 ahead of N36, i.e. a coin flip that the declared rule settles). Final after N49.
- **N49 LR pilots (umarmadbi, 15 min, seed 0):** 5e-4 1.4061, 1e-3 1.3774, **2e-3 1.3577**, 4e-3 1.3710. Best 2e-3 is interior. Runner-up 4e-3 is +0.0132 < 2.8 sigma15 (0.0174), so by the declared close-call rule (D09, applied to every size in E01) both get seeds 1, 2 and the pick is the lower 3-seed mean (4 runs, ~$1.1). Gap considered: the expected gain is small (E01: every such call ended within noise), but skipping it would give N49 less tuning than N7-N36 received and D19 promised "the declared protocol". Pattern so far fits E01: widths 512-640 prefer 2e-3 to 4e-3, with the optimum drifting down with width (parabola vertex N36 2.9e-3, N49 2.1e-3, N65 1.7e-3).
- **Then:** N49 at the pick, 60 min, seed 0 (`configs/E02/N49.yaml`; placeholder already 2e-3, change to 4e-3 only if 4e-3 wins and commit). It joins P1: within 2.8 sigma60 of the best seed-0 size -> seeds 1, 2; final call on 3-seed means.

### D21 (2026-11-04, 01:30): four workspaces; N49 60-min seeds 0, 1, 2 launched together
- **Workspaces:** jinkyungjeon (Jinkyung, $28.80) and giraffes573 (Hyewon, $28.91) verified (data 13/13, L4 OK). With joshuahanrc (~$19) and umarmadbi ($2.98) about $80 is left. umarmadbi is nearly empty: from now on it is used only for recovering its own runs and for rescoring its own checkpoints (E01 N26 s0, E02 N7/N65 s0, N26 s1/s2), which live on its Volume.
- **N49 seeds 0-2 at once:** under D19/D20, N49 seed 0 decides whether N49 enters the close-call set (within 2.8 sigma60 = 0.0161 of N36's 1.1888, i.e. <= 1.2049), and only then gets seeds 1, 2. Modal has been queueing runs for hours on some workspaces, so waiting for seed 0 costs a full round. Seeds 1, 2 are launched together with seed 0. This does not change any rule: step 1 still uses seed 0 only; if N49 falls outside the set, its seeds 1, 2 are reported as extra information and not used in the decision. Cost of that case: ~$2 of a ~$80 pool. Launched on jinkyungjeon at the LR picked by the D09 close-call rule (lower 3-seed mean of 2e-3 vs 4e-3).

### D22 (2026-11-04, 10:50): Phase 2 closed - optimal range N36-N65; N49 is the baseline size
- **N49 (jinkyungjeon, commit 0b7dde4, clean, speed -2.4 to -3.8 % vs its pilots):** s0 1.1936, s1 1.1961, s2 1.1937; mean **1.1945** (sd 0.0014).
- **P1 final (declared rule):** seed-0 close-call set {N36, N65, N49} (N49 +0.0048 < 0.0161). 3-seed means: **N49 1.1945, N65 1.1951, N36 1.1959**; gaps 0.0006 and 0.0015 vs 1.6 x pooled sigma60 = 0.0092 (pooled 0.0057 on 10 df). **Result: optimal range {N36, N49, N65}** = 36-65M parameters, 3.8-11 tokens per parameter. The curve is flat on top: three sizes spanning a factor 1.8 in parameters agree within 0.0015 BPB.
- **P2:** lowest mean is N49, an interior size: no edge run. (N92 not run; whether the plateau extends above 65M is untested: limitation.)
- **P9 (declared D18): baseline for Phases 3-6 = N49** (N26 not in the range; lowest mean). Practical bonus: N49's 3 seeds were scored on all four validation sets, so they are the FineWeb-Edu arm of Phase 3 at no extra cost. N49 at peak LR 2e-3, ~290M tokens per hour (~6 tokens/param).
- **Diagnostics:** P3 whole-range parabola ~70M (leave-one-out 57-96M, beyond the grid edge in most fits); post-hoc local fit through N36/N49/N65 ~51M, ~5.5 tokens/param. P7: the time-optimum moved from ~17M at 15 min to ~50-70M at 60 min, i.e. N_opt ~ time^0.8-1.0, steeper than Chinchilla's compute^0.5 (Kaplan 0.73). Caveats for the report: only two horizons; equal time is not equal FLOPs (MFU 0.18 -> 0.24, so bigger models received up to ~35 % more FLOPs); the optimum is flat, so its location is loosely determined.
- **Same ranking on other text:** FineWeb, DCLM and Wikipedia validation give the same picture (N36/N49/N65 lowest, small sizes worst).
- **Headline for the report:** under a fixed 60-min L4 budget the best models train on ~4-11 tokens per parameter, far fewer than Chinchilla's ~20; the 20-tokens/param model (N26) is clearly worse (5-seed mean 1.2123, +0.018). Explanations to discuss: (1) equal-time favours big models via MFU; (2) small-scale optima differ from Chinchilla's fitted constants (Porian et al. 2024, Besiroglu et al. 2024); (3) an 8k vocabulary changes the embedding share; (4) our horizon is short and the cosine schedule ends at the cap.
- **Phase 2 cost:** 9 + 4 + 3 sixty-minute runs and 8 N49 pilots ~ $15 ledger (~$18 real).

### D23 (2026-11-04, before 11:16; was stamped 11:40, an estimate, corrected in D25): Phase 3 design (E03 data quality, E04 repetition), fixed before any run; random nested subsets
- **E03:** N49 trained on FineWeb and on DCLM, seeds 0-2 each; FineWeb-Edu arm = E02-N49 s0-2. Decision on AVG4 (declared 30 Oct), real if > 1.6 x pooled sigma_avg4 (E02 0.0055 on 9 df, pooled with the new arms). 13-gram contamination check before interpretation, with a sensitivity re-score if > 1 % of a validation set overlaps a training corpus. Each arm split over two workspaces. Full design: `docs/experiments/E03_data_quality.md`.
- **E04:** N49 on random nested FineWeb-Edu subsets of 290M/k tokens, k = 2, 4, 8, 16, 32 (seed 0), vs unique-data N49 s0-2; worse if > 0.0132 (single vs 3-seed mean); boundary k confirmed with seeds 1, 2. Full design: `docs/experiments/E04_repetition.md`.
- **Gap found: the existing subset option was not random.** `train_tokens_limit` takes the first N tokens of train.bin, which is written in source-file order (D02): the first ~0.9B FineWeb-Edu tokens all come from source file 000, the same file the validation and test documents were sampled from. A prefix subset would differ from the full data in source composition (FineWeb sample files may be grouped by crawl), confounding repetition with distribution shift, and contradicting DECLARATIONS section 3 ("nested random subsets"). **Fix:** new config field `train_subset_tokens`: whole windows chosen by a fixed permutation (key 1234, independent of the run seed), so subsets are uniform over the split and nested. Unit test added (13/13 local tests pass, CPU, torch 2.15.0). Training for configs without the field is unchanged (the code path only activates when the field is > 0; `unique_train_tokens` keeps its meaning).
- **Why k = 2-32 and single seeds:** the plan's 1-16 range starts at the unique-data reference (k = 1 is what the baseline already does, 0.12 passes), and 32 shows the strongly-repeated regime Muennighoff et al. describe; a dose-response over 5 points answers "where does it start to hurt" better than 3 seeds at 2 points; the boundary gets 3 seeds (R2).
- **Workspaces:** E03 on giraffes573 and jinkyungjeon (arms split), E04 on joshuahanrc. Cost ~$11-13.
- **Not done now:** mixtures (SmolLM2-style), a repetition series on DCLM, LR re-tuning per corpus. Candidates if results and budget justify.
- **Still to build, before interpreting E03 (not before launching):** the 13-gram contamination check and a rescoring tool (also needed for N26 seed 0 and for the final test evaluation).

### D24 (2026-11-04, before 11:16; was stamped 12:10, an estimate, corrected in D25): why the optimum is ~5 tokens/param, not ~20 - post-hoc analysis
- **Context:** why is the 60-min optimum (3.8-11 tokens/param) so far from Chinchilla's ~20?
- **Is it just our small compute? No.** Plugging our budget (~9e16 FLOPs) into Chinchilla's own fitted law gives N ~27M and ~20 tokens/param (Hoffmann et al. Approach 3) or ~24M and ~27 (Besiroglu et al. 2024 refit). Their laws predict N26, which we found clearly worse.
- **Is it just equal time vs equal FLOPs (MFU)? Mostly no.** New post-hoc diagnostic `code/fit_scaling_law.py`: Chinchilla-style fit L = E + A/N^a + B/D^b to our 14 complete runs (7 sizes x 15 and 60 min). Fit: a = 0.16, b = 0.84 (Chinchilla: 0.34, 0.28); max residual 0.013 BPB. **At EQUAL FLOPs** (our mean 60-min budget) its optimum is ~56M and ~4.8 tokens/param (leave-one-point-out 40-60M, 4.2-9.6). So the MFU advantage of big models (up to ~35 % more FLOPs) is real but explains only a small part.
- **What the fit says:** at our scale the loss falls fast with more tokens at first and then levels off (large b), while it keeps falling slowly with model size (small a). After a few hundred million tokens, extra text helps less than extra parameters.
- **Plausible reasons, to discuss in the report (not tested separately):** (1) extrapolation: Chinchilla's smallest budgets were ~6e18 FLOPs, ~60x ours, and the power-law form may not hold this far down; (2) data quality: FineWeb-Edu is much denser than MassiveText, and DeepSeek LLM (2024) reported that higher-quality data shifts the optimal allocation towards bigger models; (3) recipe differences at small scale (per-size LR tuning, AdamW settings, warm-up, 8k vocabulary) shift fitted laws (Porian et al. 2024); (4) equal time favours big models a little (MFU).
- **Size of the effect in loss terms:** the valley is flat; using Chinchilla's 20 tokens/param (N26) costs only +0.018 BPB (~1.5 %), which is why a moderate change in setup can move the optimum a long way.
- **Caveats (to state with the numbers):** 14 points vs 5 parameters; only two training lengths per size; 15-min runs have a larger warm-up/compile share; the law's form is assumed. Diagnostic only; no decision depends on it.

### D25 (2026-11-04, 11:20-12:30): review before Phase 3 launch; Phase 3 rules tightened before any E03/E04 run
- **Context:** a full review of the plan and of Phases 0-2 before starting Phase 3, looking for gaps.
- **What was checked and holds:** DECLARATIONS, plan, D01-D24, E00-E04 records, code (`train.py` loader and subset,
  `bpb.py`, `prepare_data.py` dedup, `summarize.py` E03/E04), configs, manifest, ledger, all E02 run files.
  13/13 local tests pass (CPU, torch 2.15.0). The subset code is correct: windows are aligned with the full-data
  windows, nested by construction, reshuffled per pass with (seed, pass). Cross-corpus exact dedup is in place
  (manifest: 0-5 train docs removed per corpus as duplicates of another corpus's eval docs). **sigma_avg4 = 0.0055
  (9 df) re-derived from the run files: correct.** D23 is not committed yet, so its rules can still be tightened
  without breaking "rules fixed before the first run".
- **Gaps found and fixes (all before any E03/E04 run, none depends on an E03/E04 outcome):**
  1. *Machine speed can fake or hide a corpus effect.* Within E02 configs the AVG4 slope on ln(tokens) is -0.131
     (residual sd 0.0047 vs raw 0.0055). The FineWeb-Edu arm ran on slow L4s (-2.4 to -3.8 %); a 4 % token
     difference is ~0.005 AVG4, over half the R1 threshold. **Fix: E03 R5**, an ANCOVA with a common slope
     (pooled within configurations), reported beside R1; a pair whose verdict changes is reported as "not robust to
     machine speed". Same adjustment shown for E04 (R5a). Code checked against numpy OLS with group dummies
     (identical slope, residual sd and df).
  2. *LR tuned on FineWeb-Edu only.* **Fix: E03 R6**, 4 pilots of 15 min (FineWeb, DCLM x 2e-3, 4e-3; ~$1.1); if
     4e-3 beats 2e-3 on the corpus's own val by > 2.8 sigma15 (0.0188), that arm also gets a 60-min 4e-3 run as a
     sensitivity. R1 keeps 2e-3 (declared).
  3. *"Substantial 13-gram overlap" was undefined, and which documents to drop was unspecified.* **Fix: E03 R3**:
     token 13-grams vs each corpus's full train split, contaminated if >= 50 % of a doc's 13-grams occur there
     (10 % and 80 % also reported); if > 1 % of a set is contaminated, the SAME documents (union over training
     corpora) are removed for every arm, so the comparison stays paired.
  4. *E04's boundary rule ("largest k not distinguishable, if followed by a worse k") breaks on non-monotonic
     seed-0 results and was not in `summarize.py`.* **Fix: E04 R2** keyed on the smallest worse k, both bracket
     ends confirmed with 3 seeds, at most one move. Implemented and tested on simulated results for every branch
     (none worse; k = 2 worse; bracket confirmed; boundary down; boundary up; non-monotonic; missing runs).
  5. *`summarize.py E03` grouped every E03 run by corpus,* so the R6 pilots would have entered the arms. **Fix:**
     arms take 60-min runs only; flagged runs print a "replace before reading R1" warning.
  6. *Which corpus Phases 4-6 use was not fixed;* picking DCLM later because it won AVG4 would be post hoc.
     **Fix: E03 R9**, Phases 4-6 stay on FineWeb-Edu (the declared objective is FineWeb-Edu test BPB).
  7. *Planned-but-conditional follow-ups should be declared before results.* **Fix: E03 R8**, a 60/40
     FineWeb-Edu/DCLM mixture arm if the two are inconclusive or complementary per set; **E04 R5b** train-val gap
     and **R6** effective data (Muennighoff R*), new script `code/fit_repetition.py` (tested on simulated runs).
  8. *Own-corpus advantage is not perfectly symmetric* (FineWeb-Edu ~ Wikipedia in style; FineWeb ~ DCLM). Property
     of the declared metric; made visible with E03 R7 (Wikipedia alone, in-domain advantage), stated in E03 gaps.
  9. *Record times:* D23, D24 and the handoff were stamped 11:40-12:10, but the files were last saved at 11:16
     and this review started at 11:20; the stamps were estimates. Corrected in the headings (here and in the log).
- **Considered and not changed:** reusing E02-N49 as the FineWeb-Edu arm (identical config; R5 handles the speed
  difference; a fresh arm would cost ~$3 for no new information); k grid 2-32 (k = 1 is the reference); single
  seeds at seed 0 for the dose-response; workspace split; the R1 thresholds; the 13-gram check after launch
  (it changes interpretation, not training).
- **Still to build (not launch-blocking):** `code/contamination.py` + `modal_app.py::contamination` (CPU, all
  eval docs vs every train split; also covers the test sets for Phase 6); `code/rescore.py` + `modal_app.py::rescore`
  (score a saved final_model.pt on any val set, optionally with documents removed; must run in the workspace that
  holds the checkpoint; also gives N26 seed 0 its three missing val sets). `evaluate_test.py` + frozen configs before
  Phase 6. Corpus-mix option only if R8 fires.
- **Plan-level notes for Phases 4-6 (no change yet, decide after Phase 3):** credits ~$76 (jinkyungjeon ~$25.5,
  giraffes573 $28.91, joshuahanrc ~$19, umarmadbi $2.98); Phase 3 needs ~$12-20. The plan still promises depth vs
  width (E02 P6), and Phase 4 should include Muon vs AdamW (each tuned) because it has a dedicated reading; the vocab
  sweep (Phase 5) needs re-tokenised data per vocabulary and is the first candidate to cut. Experiments should end by
  ~10 Nov to leave three days for the report.
- **Affects:** E03, E04 rules and summaries (DECLARATIONS Amendments rows updated in place: not yet committed, no
  run yet); no existing result changes.
- **Built the same day (D25, after the Phase 3 files were ready for commit):** `code/contamination.py`
  (+ `modal_app.py::contamination`, 3 parallel CPU jobs, results saved on the runs Volume first so a dropped
  connection loses nothing; covers val AND test of all four sets), `code/rescore.py` (+ `modal_app.py::rescore`,
  one L4 per run, a few minutes, reports the difference to the stored score as a reproducibility check),
  `summarize.py E03` R3 section (reads results/contamination.json and results/rescore/*__nocontam.json). Tests:
  15/15 (two new: contamination = brute-force lookup incl. chunk edges; rescore reproduces the stored BPB exactly on
  CPU and document removal = hand-rebuilt file). Speed measured here: ~10 s per 50M train tokens against 10M
  evaluation 13-grams, i.e. ~10 min per training corpus. Kept out of `code/` until the Phase 3 launches were done,
  because `launch` refuses untracked files in `code/`.

### D26 (2026-11-04, 15:00-15:40): Phase 3 first results; declared follow-ups fired; E04 surprises -> two follow-ups declared before their runs
- **Runs (commit 4fce312, launched 11:54-12:00):** E03 FineWeb s0-2, DCLM s0-2 (giraffes573 / jinkyungjeon, starts 12:04-13:04),
  4 R6 pilots (giraffes573, 12:29-13:22), E04 k = 2-32 s0 (joshuahanrc, starts 12:27-13:52). All launch terminals lost
  their connection (`getaddrinfo`, 14 'failed' ledger rows, to annotate); all 15 runs completed (1 attempt, cold compile,
  1 run per container, speed -5.5 % to +0.5 % vs pilots) and were recovered ~15:00. Contamination check ran on giraffes573
  (3 CPU jobs, 283-419 s each); its local fetch first failed because the laptop's launch-only environment has no numpy
  (our bug: the merge step imported numpy at module level; nothing had started; fixed by `pip install numpy`, and now
  numpy is imported lazily).
- **E03 R1 (declared): DCLM is best.** AVG4 3-seed means: DCLM 1.3673, FineWeb-Edu 1.3954, FineWeb 1.3969; pooled
  sigma_avg4 0.0055 (13 df), threshold 0.0088. DCLM vs FineWeb-Edu +0.0280 REAL, DCLM vs FineWeb +0.0296 REAL,
  FineWeb-Edu vs FineWeb +0.0016 inconclusive. **Prediction missed** (we expected FineWeb-Edu or DCLM within ~0.01).
- **Robustness (declared):** R5 speed-adjusted: same verdicts (DCLM +0.0291, +0.0309; the DCLM runs had the slowest
  machines, -2.1 to -5.5 %). R6: 2e-3 vs 4e-3 differ by 0.0017 (FineWeb) and 0.0012 (DCLM) on own val -> consistent.
  R3: union contamination FineWeb-Edu val 3.38 %, FineWeb 1.50 %, DCLM 1.26 %, Wikipedia 0.25 % -> sensitivity
  re-score REQUIRED (commands prepared ~15:05; launch time to be logged). The largest own-corpus near-duplicate share is FineWeb-Edu's
  (2.83 % of its val docs have >= 50 % of their 13-grams in FineWeb-Edu train; DCLM 0.42 %), so contamination, if
  anything, favours the FineWeb-Edu arm, not DCLM.
- **R7 (descriptive):** DCLM is also lowest on Wikipedia (1.4490 vs 1.4616 FWE, 1.4744 FW; 0.0126 vs a per-set noise
  scale of 0.0117) and equal to FineWeb-Edu on raw FineWeb val (1.3819 vs 1.3850). FineWeb-Edu wins only on its own
  text, by a lot (0.07-0.08). In-domain advantage: DCLM -0.164, FineWeb-Edu -0.078, FineWeb -0.051. Post-hoc
  descriptive (labelled as such): on the sets neither arm trains on (FineWeb, Wikipedia), DCLM <= FineWeb-Edu on both.
  Plausible reading for the report: the educational filter removes whole genres (forums, Q&A, how-to, product pages)
  that DCLM's classifier (trained on instruction/ELI5-style positives) keeps, so a FineWeb-Edu model is narrow.
- **R9 holds (declared before results):** Phases 4-6 stay on FineWeb-Edu: the declared objective is FineWeb-Edu test,
  where FineWeb-Edu is best by ~0.08. The DCLM result is a separate finding about breadth.
- **R8 fired (declared trigger):** FineWeb-Edu and DCLM are complementary per set (FWE wins FWE val, DCLM wins DCLM
  and FineWeb val beyond noise). Note: with an own-corpus advantage this second condition almost always fires; it was
  declared that way, so the mixture is run. **Rules fixed now, before its runs:** `configs/E03/N49_mix_fwe60_dclm40.yaml`,
  60/40 by tokens, seeds 0-2; each corpus contributes a 1B-token random pool (nested-subset key; the full splits of two
  corpora would not fit the 16 GB container; 1B is 6-9x what an hour reads, so nothing repeats); judged by R1 against
  both parents (pooled sigma incl. the mix arm); R5 alongside; secondary: FineWeb-Edu val vs the FineWeb-Edu arm
  (1.6 sigma60 = 0.0091). Re-scored without contaminated docs like every arm. **Prediction:** AVG4 within noise of
  DCLM and clearly better than FineWeb-Edu; FineWeb-Edu val ~0.02-0.03 worse than the FineWeb-Edu arm.
- **E04 R1 seed 0:** deltas vs unique data k = 2 +0.0099 (not distinguishable), 4 +0.0245, 8 +0.0988, 16 +0.470,
  32 +2.448 (all WORSE). R2 bracket {2, 4} -> seeds 1, 2 (joshuahanrc; commands given ~15:05, launch time to be logged). **Prediction
  missed:** we expected 2 and 4 passes to be indistinguishable (Muennighoff et al.). Validation peaks at 6.9-7.9 passes for
  k = 8, 16, 32, then rises while training loss keeps falling (k = 32: train 0.23 nats/token, val 3.64 BPB, worse than the
  untrained model's 3.46): memorisation. Train-val gap vs reference: +0.02 (k=2), +0.18 (4), +0.55 (8), +2.3 (16), +9.6 (32).
- **Flag note:** k = 32 is flagged UNSTABLE by the P10 rule (a training-loss jump of 0.149 between 30-s logs). Gradient
  norms stay < 1 and only 0.25 % of steps are clipped; the jump is the per-pass sawtooth of a memorising model, not an
  optimisation instability. Kept (rule: flags are kept), explained.
- **R6 effective data (diagnostic):** with the full 14-point loss law (reproduces D24: 56M, 4.8 tokens/param), the 2nd
  pass was worth ~0.74 of new data (k=2: 253M effective of 291M seen) and k=4 69 %; effective data FALLS beyond 8
  passes. Muennighoff's form only saturates, so its fitted R* (0.6, LOO 0.3-1.3) is not meaningful here; reported with
  that caveat. Calibration of the inversion on the unique runs: 0.99-1.04x.
- **Why so different from Muennighoff et al.? Two candidate causes, each tested by a follow-up declared NOW, before any
  run (amendment policy item 3):**
  - **F2, model too large for the data:** at 8 passes N49 has 0.74 unique tokens per parameter. Muennighoff et al. find
    excess parameters lose value faster than repeated data (R*_N ~5.3 < R*_D ~15.4), i.e. with limited data a smaller
    model should win. Test: N17, N26, N65 (E01 LRs) on the SAME 36.25M-token subset, 60 min, seed 0 (`configs/E04/*_U36M.yaml`).
    In 60 min they read it ~21, ~14, ~7 times. Each vs E04-N49_rep8 s0: real if |diff| > 2.8 sigma60 = 0.0161.
    Also reported: each size's "cost of the limit" vs its own unique-data E02 mean. **Prediction:** N17 and N26 better
    than N49 (less memorisation despite more passes), N65 worse. This is the project question ("where should the hour
    go?") under a data constraint.
  - **F3, exact repetition of identical samples:** our loader shows the same 513-token windows every pass; common
    pipelines re-cut the text each epoch. Test: N49 at k = 8 with window boundaries moved every pass
    (`train_subset_shift`, new `ShiftedSubsetLoader`; seams between subset windows marked with EOT, +0.2 % tokens),
    seed 0, vs E04-N49_rep8 s0, same threshold. **Prediction:** modestly better (0.01-0.03), far from closing the +0.099 gap.
  - Not tested (limitations): regularisation (dropout, higher weight decay), LR re-tuning for repeated data.
- **Code (D26):** `train.py` gains `ShiftedSubsetLoader` and `MixLoader` (+ `corpus_mix`, `mix_pool_tokens`,
  `train_subset_shift` config fields); the existing paths are untouched (tiny full-data and subset runs give bit-identical
  weights with the committed and the new train.py). Tests 17/17 (new: shifted windows = same text, boundaries move, each
  stream token a target at most once per pass, crash-resume exact; mixture shares exact to one window, pools, crash-resume
  exact, bad configs refused). `summarize.py`: E04 main table limited to N49 fixed-window runs (the F2 runs use the same
  subset and would have been mixed into k = 8), follow-up table; E03 mixture line. `modal_app.py rescore --fetch-only`.
- **Cost:** Phase 3 so far 12 x 60-min + 4 pilots ~ $11 ledger; follow-ups: E04 R2 4 runs, F2 3, F3 1, mixture 3, rescore
  ~9 short L4 jobs ~ $12. Remaining after that ~$48 for Phases 4-6.
- **Update 17:20 (declared rules applied, no new decision):** E04 R2 3-seed means: k = 2 +0.0066 (not distinguishable), k = 4 +0.0244 (worse) -> boundary between 2 and 4 passes, confirmed; no move. E03 R3 (required): AVG4 without the 145 contaminated val docs (same for every arm): DCLM vs FWE +0.0290, vs FineWeb +0.0298 (REAL; with all docs +0.0280 / +0.0296), FWE vs FineWeb +0.0007 inconclusive -> E03 conclusion unchanged. Re-scores equal the stored BPB exactly.

### D27 (2026-11-05, 00:30-01:15): E03 mixture and E04 F2/F3 results; one extra reference seed proposed; summary speed-check bug fixed
- **Runs (commit 3f371ee = D26, launched 17:22-17:24 on 4 Nov):** mixture s0, s1 + N17, N26 F2 (giraffes573), mixture s2 + N65 F2
  + F3 shift (jinkyungjeon). All launch terminals lost their connection (7 'failed' rows); all 7 runs completed (1 attempt, cold
  compile, 1 run per container, clipped 0.1-0.3 %) and were recovered ~00:20; mixture re-scores without contaminated documents
  done (exact match to stored BPB). Realised token shares 0.600/0.400 in every mixture run.
- **E03 mixture (declared R1):** AVG4 3-seed mean **1.3460** (s0 1.3424, s1 1.3460, s2 1.3496): better than DCLM by **0.0213**,
  than FineWeb-Edu by 0.0494, than FineWeb by 0.0509; pooled sigma_avg4 0.0053 (15 df), threshold 0.0085: **all REAL**. Robust:
  R5 speed-adjusted 0.0191 over DCLM (REAL); R3 without contaminated docs 0.0208 (REAL).
  Per set: FWE val 1.2167 (FWE arm 1.1945: +0.0222, beyond 1.6 sigma60 = 0.0091), FineWeb val 1.3713 (better than both
  parents: FWE 1.3850, DCLM 1.3819), DCLM val 1.3811 (DCLM arm 1.3622: +0.0189), **Wikipedia 1.4150, lowest of all arms by
  0.034** (DCLM 1.4490, FWE 1.4616; per-set noise scale 0.0117).
  **Prediction partly missed:** we expected AVG4 within noise of DCLM; the mix is clearly better than both parents. The FWE-val
  cost (+0.022) matched the prediction (0.02-0.03).
  Reading: the mix gives up a little on each parent's own text but gains on text neither covers (Wikipedia, raw web) more than
  either parent, i.e. the two filters keep complementary content (consistent with SmolLM2's motivation for mixing them).
  **R9 unchanged:** Phases 4-6 stay on FineWeb-Edu (declared objective; FWE arm is best on FWE val by 0.022). Reported as the
  main practical recommendation for a broad objective.
- **E04 F2 (declared, single seed vs E04-N49_rep8 s0 1.2933, threshold 0.0161):** N17 1.3036 (+0.0103, inconclusive, 22.0
  passes), N26 1.2913 (-0.0019, inconclusive, 13.7 passes), **N65 1.2688 (-0.0245, BETTER, 6.6 passes)**. **Prediction missed
  (opposite direction):** we expected smaller models to win under a data limit; at equal TIME the bigger model reads fewer
  passes and does at least as well. Validation minimum moves to later passes for smaller models (N17 19.8, N26 10.9, N49 6.9,
  N65 5.9 passes): smaller models tolerate more repeats before overfitting but still end no better.
  "Cost of the limit" (vs each size's unique-data E02 mean): N17 +0.074, N26 +0.077, N65 +0.074, N49 +0.099, N49-shift +0.077.
- **E04 F3 (declared):** shifted windows 1.2709 vs fixed 1.2933: **-0.0223, BETTER**. Showing the identical samples every pass
  accounts for ~1/4 of the 8-pass damage (0.022 of 0.099). Prediction (0.01-0.03) held.
- **Gap found in the evaluation: the reference is one seed.** Every F2/F3 verdict is against E04-N49_rep8 seed 0 alone, and its
  cost of the limit (+0.099) is the outlier among five runs that otherwise cost +0.074-0.077. A seed-0 value 0.02 above its
  configuration's mean is ~4 sigma60 if sigma60 applies, so probably not noise alone, but repeated-data runs could be noisier
  and the k = 8 point is also the single most important point of the E04 dose-response. **Proposal (declared now, before the
  runs; labelled as added after F2/F3):** E04-N49_rep8 seeds 1, 2 (joshuahanrc, ~$2.4). F2/F3 are then reported both against
  seed 0 (declared rule) and against the 3-seed mean (threshold single vs 3-seed mean 2 sigma60 sqrt(4/3) = 0.0132); if the two
  disagree, the follow-up result is reported as "sensitive to the reference seed". The k = 8 point of the main E04 table then
  becomes a 3-seed value (the R2 rule does not change: k = 8 was not in the bracket).
- **Bug found and fixed (no result changes):** the per-run speed check in `summarize.py` E03/E04 compared every run with N49's
  pilot speed (written when only N49 ran in Phase 3). It flagged E04-N65_U36M as "SLOW MACHINE -> replace" although its speed
  is -0.1 % vs N65's own pilots (N17 +11.4 %, N26 -4.5 %). Now uses each run's own size. All Phase 3 runs pass.
- **Ledger:** the 21 'failed' rows from the 4 Nov 11:54-12:00 and 17:22-17:24 launches annotated (connection dropped; run
  completed and recovered -> row named).
- **Cost:** Phase 3 total ~ 19 x 60-min + 4 x 15-min + 7 x 60-min follow-ups + CPU/rescore. Credits (Modal dashboards, 17:25 on 4 Nov):
  umarmadbi $2, joshuahanrc $8, jinkyungjeon $21, giraffes573 $23, before the 7 follow-ups (~$8.5).

### D28 (2026-11-05, 01:00-02:00): Phase 4 designed (E05 Muon vs AdamW, E06 depth vs width), fixed before any run
- **Context:** Phase 4 was designed while the E04 reference seeds run; credits ~$45 after Phase 3, deadline
  13 Nov (experiments by ~10 Nov). Time, not money, is now the main limit.
- **Choice of Phase 4 questions:** (1) **Muon vs AdamW** (plan item; dedicated reading in the project folder); (2) **depth
  vs width at N49's size** (promised in the plan's Phase 2 and E02 P6, never run). Not chosen: WSD vs cosine (with a
  fixed time horizon the literature expects a tie; little to learn), batch size (fixed below the critical batch in E00).
- **E05:** torch.optim.Muon on the 44 hidden block matrices (43.8M params), AdamW (2e-3, as baseline) on embeddings and
  LayerNorm; RMS-matched update scaling (Liu et al. 2025) so LR/weight decay are on AdamW's scale; Muon LR grid 1e-3 ...
  8e-3 x 15 min (same protocol as AdamW in E01); then 3 seeds x 60 min vs E02-N49. Real if > 0.0091. Muon finals are
  checked for slow machines against Muon pilots (gap found while designing: the generic check would flag every Muon run
  as slow, because Muon is slower by design). Prediction: Muon better by 0.01-0.02.
- **E06:** 6 x 768 and 19 x 448 (49.2M, 49.7M vs 48.8M), 2-LR pilots each (reduced protocol; E01 width rule as
  default), seed 0 vs N49 mean (0.0132), close calls get seeds 1, 2 (0.0091). Prediction: no shape clearly better.
- **Code:** `MuonAdamW` wrapper in train.py, config fields `optimizer`, `muon_lr`, `muon_momentum`, `muon_ns_steps`,
  `muon_adjust_lr`; the schedule loop multiplies each group's lr by its `lr_scale` (1.0 for AdamW groups: bit-identical
  AdamW path, tested against D26 code on full and subset runs). New test (18/18): Muon gets exactly the block
  matrices, parameters fully covered, lr ratio, learns, exact crash-resume, bad configs refused. `summarize.py E05|E06`
  (tested on simulated runs: close call, edge, speed check vs Muon pilots, seed-0 close call).
- **Phase 4 cost:** E05 ~$5-6, E06 ~$3.6-8.4. Remaining for Phase 5 (vocab, reduced) and Phase 6 (final + test) ~$25.

### D29 (2026-11-05, 10:05-10:30): E04 reference seeds; Phase 3 closed
- **Runs:** E04-N49_rep8 seeds 1, 2 (joshuahanrc, commit a1dd2b9 = D27; launched 00:44, starts 00:52 and 02:38; clean: 1
  attempt, cold compile, speed -3.2 / -1.0 %). Launch terminal lost its connection (2 'failed' rows, annotated); recovered
  ~10:00.
- **k = 8, 3 seeds:** 1.2933 / 1.2722 / 1.2873, mean **1.2842** (+0.090 vs unique; seed 0 was the worst, so the +0.099 of
  D26 was partly seed luck). Cost of the 36M-token limit for N49 is now +0.090 (other sizes +0.074-0.077).
- **F2/F3 vs the 3-seed reference (declared in D27; threshold 0.0132):** N17 +0.0193 WORSE, N26 +0.0071 inconclusive,
  N65 -0.0155 BETTER, shifted windows -0.0133 BETTER (just above the threshold). Against seed 0 (declared D26 rule): N65 and
  shift BETTER, N17/N26 inconclusive. The two references agree on N65 and shift (D27 rule: not "sensitive to the
  reference seed").
- **Gap found: repeated-data runs are noisier.** Seed sd at k = 2, 4, 8: 0.0048, 0.0048, 0.0108; pooled 0.0074 (6 df) vs
  sigma60 0.0057 (10 df); F = 1.68, one-sided p = 0.22, so not significant with these few runs, but plausible
  (memorisation dynamics). **Sensitivity (reported, not decisive):** with the repeated-data sd the threshold becomes
  0.0171, and N65 (-0.0155) and shift (-0.0133) become inconclusive; N17 stays WORSE. **Reporting decision:** the declared
  verdicts stand, stated with this sensitivity: "N65 and re-cut windows are better by ~0.013-0.016, at the edge of our
  resolution"; the robust findings are (a) a smaller model does not help under a data limit (N17 worse, N26 no better),
  (b) repetition damage is large and only partly (<= ~1/4) due to exact sample repetition.
- **More seeds considered and NOT run:** +2 seeds each of N65_U36M and N49_rep8_shift (~$4.8) would settle the two
  edge verdicts. Not worth it: credits ~$41 must cover Phase 4 (~$9-14), Phase 5 (~$5-7), Phase 6 (~$5) and a reserve;
  neither verdict changes a later phase (Phases 4-6 use unique data). Stated as a limitation.
- **Bug fixed (no reported number affected):** `fit_repetition.py` loaded every E04 run, so after D26 it would have fed
  the F2 runs (other sizes) and F3 (shifted) into the N49 effective-data fit. Now only N49 fixed-window runs. With 3
  seeds: effective / seen tokens k = 2 0.91, 4 0.70, 8 0.38 (16: 0.09, 32: 0.01, single seeds); fitted R* 1.1 (LOO
  1.0-1.8), not meaningful because loss rises past ~8 passes. The D26 numbers were computed before F2/F3 existed.
- `summarize.py E04` now prints the follow-ups against the 3-seed reference with both thresholds.
- **Phase 3 is closed.** Remaining Phase 3 items for Phase 6: test-set scoring of the E03 arms (all four sets, incl. the mixture).

### D30 (2026-11-05, 10:15-10:45; review work from ~01:30): review of the plan and Phases 0-3 before Phase 4; Phase 4 rules revised before any Phase 4 run
- **Context:** a review of the plan and everything done before starting Phase 4, with attention to gaps.
  Written in parallel with D29 (which closed Phase 3 at 10:05-10:30); merged onto D29's files (its
  `summarize.py` E04 change and `fit_repetition.py` kept; E01/E03/E04 summary output byte-identical before and after the
  merge).
- **Plan and Phases 0-3: hold up, nothing to re-run.** Declared budget, data, tokenizer, metric, seeds, noise rule and
  amendment policy have been applied consistently; every comparison's rules were recorded before its runs; failures and
  missed predictions are recorded. Open items that are NOT Phase 4 but must not be forgotten: (1) the test lock
  (`code/evaluate_test.py`, `configs/frozen_configs.yaml`) must be built and committed before any test scoring;
  (2) Phase 5 needs re-tokenised data per vocabulary on the workspace that runs it, a CPU job of ~0.5-1 h, so Phase 5
  should be designed and its data prepared WHILE Phase 4 runs (time, not money, is the limit: experiments end ~10 Nov);
  (3) Phase 6 needs a declared rule for combining winners (e.g. a REAL Muon win and a REAL shape win were never tested
  together, and Muon's LR is tuned at 11 x 576 only), to be fixed before Phase 6; (4) the original project plan
  (30 Oct) is out of date (Phase 4 contents, reduced Phase 5); the status summary is current; (5) authors.md, invite
  alexxthiery, page limit 8 vs 10.
- **Phase 4 choice re-checked:** Muon (plan item, own reading) and depth vs width (topic page asks it; promised in
  Phase 2) kept. Plan's other Phase 4 items: bf16/compile throughput is already answered by E00 (compile +19 % tokens/h;
  reported as an efficiency result); WSD vs cosine dropped (fixed, known horizon: little to learn); batch size dropped
  (E00: a larger micro-batch gives no speed, so a larger batch only means fewer steps; 32k is far below the critical batch,
  where a smaller batch is not expected to help much either). Reasons stated in the report.
- **PyTorch's Muon source read (torch 2.15, `torch/optim/_muon.py`):** quintic Newton-Schulz in bf16 after Frobenius
  normalisation; Nesterov momentum via lerp; `match_rms_adamw` scale 0.2 sqrt(max(m, n)); weight decay
  `p *= 1 - lr x wd` with the Muon group's own LR (coupled like AdamW's); eager, one matrix at a time. Matches D28's
  description. Overhead estimate: ~585 GFLOP of Newton-Schulz per step vs ~8.6 TFLOP of training per step (~7 %), on an
  L4 that runs power-capped (72 W, SM ~870 of 2040 MHz in every run), so these FLOPs cost time almost one-for-one.
- **Gaps found in the D28 Phase 4 design and code (nothing had run):**
  1. **E05 had no horizon check, AdamW did.** The declared protocol (sec. 1, D07) checks the 15-min ranking at 60 min for
     the baseline; AdamW's ranking was checked (N26), Muon's would not have been, and the optimal LR can shift with run
     length differently for a different optimiser. -> R1b: D07 applied to Muon (60-min seed 0 at the pick, the runner-up,
     and the pick's lower neighbour if the runner-up is the higher one; contradicted only if another LR wins by >
     2.8 sigma15). Extra cost 1-2 runs (~$1.2-2.4). **R1c fairness:** if Muon's check is contradicted, AdamW N49 gets
     the same check at 1e-3 and 4e-3 (AdamW at N49 inherited N26's check), so neither optimiser ends better tuned.
  2. **E06 used 2 LRs instead of the declared 4**, with an E01 "width rule" fallback. With two values the best is always
     at an edge and nothing tests 1e-3 for the wide model or 8e-3 for the deep one, so a shape difference could be an LR
     difference. -> the full declared protocol (5e-4 ... 4e-3, edge, close calls), the grid every size had. +4 pilots (~$1.2).
  3. **Bug: the edge rule only looked at the original grid ends** (`_pick_lr`), so after one extension it could not fire
     again (D07 allows two). -> judged on the LRs actually run, at most 2 per side.
  4. **Bug: E06 finals had no slow-machine check** (`_run_checks(r, {})` gives no speed reference). -> each shape vs its
     own pilots.
  5. **Bug: a run flagged "SLOW MACHINE -> replace" was still counted** in the E05/E06 decision. -> excluded until replaced;
     with 2 of 3 seeds no verdict is printed.
  6. **Placeholder final configs** (`N49_muon.yaml`, `W6x768.yaml`, `D19x448.yaml`, LR edited by hand after the pilots) ->
     one 60-min config per LR (`*_lr<x>_60min.yaml`), so nothing is edited between pilots and finals and no extra commit is
     needed; `summarize.py` prints which files to launch. Placeholders moved to `docs/_stage_d_removed/`.
  7. **"Per step vs per time" was promised but not computed.** -> R4a from the learning curves (both arms, all seeds):
     difference at equal time and at equal steps; the gap is what the slower (or faster) steps are worth. A dedicated
     shortened-budget AdamW control was considered and rejected: one seed has noise ~0.006, the size of the effect. The
     machine-speed slope (E02 within-size, -0.07 +- 0.04 BPB per ln tokens) is too imprecise to do this, so it is used only
     for the R4b sensitivity.
  8. Also added (reported, never decisive): R4b machine-speed sensitivity (84 % of seed noise is machine speed, E02);
     R4c AVG4 + per set; R4d clipping/tokens/MFU; R4e the same comparison at 15 min (training-duration question of the
     Muon reading); LR sensitivity curves of both optimisers / all shapes.
- **Considered and not changed:** splitting the fused qkv matrix for Muon (modded-nanogpt does) - needs our own Muon or a
  changed model, and a 15-min ablation cannot resolve the small expected effect; disclosed as a limitation. Tuning Muon's
  momentum/NS steps or the AdamW part of the Muon arm - not done for AdamW either; disclosed. A fresh concurrent AdamW
  control seed - the speed checks cover drift; not worth $1.2. Batch size with Muon - out of scope.
- **Tests:** 18/18 (torch 2.15 CPU). All 26 E05/E06 configs load with the declared values (LR, budget, optimizer,
  eval sets); the AdamW fairness configs differ from E02-N49 only in run_id and lr. Parameter counts 48.82 / 49.16 /
  49.68M re-checked by building the models. `summarize.py E05|E06` exercised on simulated results for every branch:
  waiting, edge (1 and 2 extensions, then limit), close call pending/decided, horizon with lower/higher runner-up,
  contradicted (incl. AdamW fairness, AdamW also contradicted), slow-machine exclusion, 2-of-3 seeds, seed-0 close call,
  3-seed verdict, equal-steps view in both directions. E01/E03/E04 output unchanged.
- **Cost and time:** E05 ~$6-9 (giraffes573, $19), E06 ~$5-12 (jinkyungjeon, $17); typical Phase 4 ~$13, worst ~$21,
  leaving >= ~$20 for Phases 5-6 (~$10-12). Order: E05 and E06 pilots together -> (close calls / edges) -> E05 finals +
  horizon and E06 seed 0 together -> E06 close-call seeds. About 4 rounds of 20-65 min plus Modal queueing: one day.
- **Prediction for the review itself (to check later):** the horizon check will not flip Muon's pick (as for AdamW at
  N26), and at least one E06 shape will need an edge or close-call pilot.

### D31 (2026-11-05, 10:55-11:30; status 12:11): Phase 5 data builder written and tested, NOT run - build deferred until Phase 4 is done
- **Status (12:11):** the build is deferred until every Phase 4 run is finished. The builder is parked
  in `docs/_stage_e/tools/` (local, not committed; copy to `tools/`, commit, then run). Cost of waiting: wall-clock only
  (~30-60 min CPU before Phase 5 runs can start). Everything below holds unchanged when it is run.
- **Update 19:05-19:15:** tools/ committed (c21d1d3) and `build` run; Modal aborted while packing the image ("tools/__pycache__/vocab_app.cpython-311.pyc was modified during build process": running the app file locally writes Python's cache into the folder
  being uploaded). Nothing started on Modal; no data written. Fix: `ignore=["**/__pycache__/**", "**/*.pyc"]` on both
  `add_local_dir` calls of tools/vocab_app.py (Modal 1.6.3 option). No logic changed.
- **Context:** to save time (experiments end ~10 Nov), build the Phase 5 vocabulary data while Phase 4 runs, on the condition that
  it is done only if no Phase 4 result can affect the build.
- **Can Phase 4 change the build? No.** The build is a tokenizer + token files. Its inputs are the raw text already on the
  Volume, the declared tokenizer recipe (DECLARATIONS sec. 4: byte-level BPE on 200 MB of each training corpus's train split)
  and a vocabulary size. Phase 4 decides the optimiser and the shape, i.e. which model is trained on this data, not the data.
  The corpus is fixed (FineWeb-Edu, R9). Context length is applied when training windows are cut, not in the token files.
  The sizes built are the declared list (4,096, 16,384, 32,768; 8,192 exists); which of them get runs, with which model,
  optimiser and LR protocol, is the Phase 5 design, made later and allowed to use Phase 4 results. Another size (e.g.
  GPT-2's 50,257) would be an extra build, never a rebuild.
- **Gap found: the reverse risk.** `code/prepare_data.py`'s tokenizer/tokenize stages write to `/data/tokenizer/`,
  `/data/tokens/` and rewrite `/data/manifest.json`, the files every Phase 4 run reads at start. Running them with another
  `--vocab_size` would have swapped the data under every Phase 4 run that had not started yet (a run loads its token files
  when it starts; ids >= 8,192 would crash or, worse, silently mis-train). -> new `tools/prepare_vocab.py`: writes only under
  `/data/vocab/v<V>/` (every path checked), builds in a temporary folder renamed only when complete, refuses to overwrite,
  reuses the Phase 0 helpers of `prepare_data.py` unchanged and the same trainer settings, checks every split has the same
  documents and bytes as the baseline. Tokenizes FineWeb-Edu train/val/test and the val/test splits of FineWeb, DCLM,
  Wikipedia (AVG4 and Phase 6 test scoring). `tools/vocab_app.py` (separate Modal app, same pinned image and data Volume):
  `build` (one CPU container per size + the 8,192 reproduction check), `fetch` (copies manifests/tokenizers to
  `data/vocab/v<V>/`, checks the reproduction, reports bytes/token and whether smaller vocabularies' merges are a prefix of
  larger ones), `verify` (hashes every variant file AND every baseline file).
- **Why tools/ and not code/:** `launch` fingerprints code/, configs/, modal_app.py, requirements.txt (`code_sha256`). Adding
  files there now would give Phase 4's later rounds a different fingerprint from their pilots. With tools/, only the git
  commit differs, training code and fingerprint are identical. The `train.py` change Phase 5 runs need (a config field
  selecting `/data/vocab/v<V>/`) is deliberately NOT written yet: it is training code and is staged after the last Phase 4
  launch (or shown bit-identical for the existing paths, as for earlier additions).
- **Checks:** `tools/test_prepare_vocab.py` (fake data): at the same vocabulary the builder reproduces the Phase 0 pipeline
  byte for byte (tokenizer.json, token_bytes.npy, every token file); baseline tokenizer, token files and manifest unchanged; no
  temporary folder left; same documents/bytes per split; refuses overwrite, a different tokenizer sample size and paths
  outside vocab/. On Modal, `fetch` must print "8,192 REPRODUCTION CHECK: PASSED" (the builder's 8,192 tokenizer = the
  recorded Phase 0 hash, proving the recipe is identical at full scale) and `verify` "baseline 13/13 unchanged" before any
  variant is used.
- **Where and cost:** giraffes573 (expected to have the most credit after E05). CPU only, 4 containers (8 CPU, 32 GB),
  estimated ~$1-1.5 and 30-60 min. A second workspace would need its own build (same tool; `verify` against the fetched
  manifests proves identical files).

### D32 (2026-11-05, 12:25-12:50): Phase 4 round 1 (LR pilots) evaluated; round 2 = close calls (+ E05 60-min seed 0 at both candidates)
- **Runs (commit 7e1b431, launched 10:35-10:37, starts 10:35-12:03):** E05 Muon lr 1e-3/2e-3/4e-3/8e-3 s0 (giraffes573), E06 W6x768 and
  D19x448 lr 5e-4/1e-3/2e-3/4e-3 s0 (jinkyungjeon). Both launch terminals lost their connection (11 'failed' rows, annotated);
  recovered ~12:20. **All 12 clean:** committed code, NVIDIA L4, 1 run per container, 1 attempt, cold compile (first
  step 29-64 s), 900 s counted, no divergence, identical validation set (2,586,692 scored tokens, 10,000,575 bytes, as AdamW).
  Speeds within -6 % of each configuration's median (no slow machine). Muon pilots: 44 matrices on Muon (as designed).
  Ledger: E05 lr1e-3 was recorded twice (launch terminal + a second recover; identical JSON, old copy in _replaced/): second row
  marked DUPLICATE, cost 0.
- **E05 (15 min, seed 0):** 1e-3 1.3086, **2e-3 1.2917**, 4e-3 1.2953, 8e-3 1.3167. R1: best interior (no edge); runner-up 4e-3
  within 0.0036 <= 2.8 sigma15 = 0.0188 -> **CLOSE CALL 2e-3 vs 4e-3** -> seeds 1, 2 of both (declared rule). Same best LR as AdamW.
  **vs AdamW at 15 min (R4e, descriptive):** Muon 2e-3 1.2917 vs AdamW 2e-3 3-seed mean 1.3508: **-0.059**, about 4x the predicted
  60-min gain. Checked, not an artifact: same validation set; Muon lower in training loss too (3.46 vs 3.61-3.65); the learning
  curves show Muon ahead at every step, **with the gap shrinking steadily** (-0.15 at step 200, -0.12 at 800, -0.07 at 1,600,
  -0.065 at 2,100): Muon's advantage is largest early, so the 15-min gain overstates the 60-min one - exactly why the horizon check
  and the 60-min finals decide. Throughput -5.0 % vs AdamW (prediction 5-12 %: held, low end). Clipped steps 2.8-7.2 % vs ~1 % for
  AdamW: all in the first steps (every run starts at gradient norm ~9.5; Muon needs more steps to fall below the clip of 1.0);
  none clipped after that, gradient norms 0.4-0.6. Not an instability (the P10 >5 % flag would fire on lr 1e-3 only, a non-contender).
  Muon's LR basin is wider: 3 of 4 values within 2.8 sigma15 of the best (AdamW: 2 of 4).
- **E06 (15 min, seed 0):** 6x768: 5e-4 1.3702, 1e-3 1.3272, **2e-3 1.3194**, 4e-3 1.3602 -> CLOSE CALL 2e-3 vs 1e-3 (gap 0.0078).
  19x448: 5e-4 1.4861, 1e-3 1.4151, **2e-3 1.3851**, 4e-3 1.3925 -> CLOSE CALL 2e-3 vs 4e-3 (gap 0.0074). No edge. **Speed vs 11x576:
  wide +15.8 % (MFU 0.26 vs 0.23), deep -12.4 % (MFU 0.21).** Prediction: wide "slightly faster" - missed (much faster); deep -5-10 % -
  about right. At 15 min (descriptive, 1 seed vs 3): wide -0.031 vs N49's 1.3508, deep +0.034. D30's prediction "at least one shape
  needs an edge or close-call pilot" held (all three configurations are close calls; the 2-LR design of D28 would have picked
  without testing 1e-3 for the wide model, which is now in the close call).
- **Round 2 (decided now):** (a) the declared close calls: E05 lr 2e-3, 4e-3 x s1, s2 (4 pilots, giraffes573); E06 6x768 lr 1e-3,
  2e-3 and 19x448 lr 2e-3, 4e-3 x s1, s2 (8 pilots, jinkyungjeon). (b) **E05 60-min seed 0 at 2e-3 AND 4e-3 now (giraffes573).**
  Reason: whichever wins the close call, both are required (the pick's seed 0 = its first final and horizon run; the other = the
  horizon runner-up), so starting them now saves a round (~1.5 h) at no extra cost. The 15-min decision cannot see them: the
  pilots finish ~45 min earlier and R1 uses 15-min runs only. The lower neighbour 1e-3 (needed only if 2e-3 wins, D07) and
  the finals' seeds 1, 2 wait for the pick. (c) **Not done:** E06 60-min runs at both candidates (~$2.4 for runs half of which
  would be unused, and it would create unplanned 60-min LR evidence for the shapes; shapes get only the declared 15-min protocol).
- **Known display issue (no rule change):** `summarize.py E05` would print the single-seed rule once the Muon pick's seed 0 is in;
  for E05 the declared R2 always needs 3 seeds. Applied by hand; the code is fixed after the last Phase 4 launch (changing code/
  now would change Phase 4's code fingerprint).
- **Cost:** round 1 ~$2.7 (12 pilots); round 2 giraffes573 ~$3.6, jinkyungjeon ~$2.4.

### D33 (2026-11-05, 15:16-15:35): Phase 4 round 2 evaluated - all LR picks made; Muon 60-min seed 0 ahead; round 3 = finals
- **Runs (commit 7e1b431, launched ~12:30; starts 12:31-13:31):** 12 close-call pilots + E05 60-min s0 at 2e-3 and 4e-3. All clean
  (committed, L4, 1 container, 1 attempt, cold compile 21-64 s, full budget, no divergence). 9 reached the laptop via the launch
  terminals, 5 recovered at ~15:15 (5 'failed' connection rows annotated). Clipping again only in the first steps.
- **R1 picks (declared close-call rule, lower 3-seed mean):** Muon **2e-3** (1.2914 vs 4e-3 1.2934); 6x768 **2e-3** (1.3178 vs 1e-3
  1.3259); 19x448 **2e-3** (1.3846 vs 4e-3 1.3933). Every configuration of Phase 4 ends at 2e-3, the AdamW N49 value.
- **15 min, 3 seeds each (R4e, descriptive; threshold 2 SE = 0.0109):** Muon -0.0593, 6x768 -0.0329, 19x448 +0.0338 vs AdamW 11x576
  (1.3508). The same ordering as seed 0 alone.
- **E05 60-min seed 0:** 2e-3 **1.1686**, 4e-3 1.1689 (equal; the horizon check needs 1e-3 too, D07, because the 15-min runner-up 4e-3 is
  the higher neighbour). Muon 2e-3 s0 vs AdamW N49 3-seed mean 1.1945: **-0.0259** (decisive rule needs 3 seeds; > the 0.0132
  single-seed scale). Learning-curve gap vs AdamW s0: -0.052 at 25 %, -0.031 at 50 %, -0.025 at 75 %, -0.027 at 100 % of the hour:
  the early lead shrinks and then holds at ~0.026 over the second half. Muon read 278M tokens (8,470 steps) vs AdamW 291M (8,869),
  steady speed -3.1 % vs its pilots. AVG4 1.3721 vs AdamW 1.3954 (-0.023). Prediction (0.01-0.02) on track to be exceeded.
- **Round 3 (as the declared rules print):** giraffes573: E05 2e-3 60 min s1, s2 (finals) + 1e-3 60 min s0 (horizon). jinkyungjeon:
  E06 W6x768 2e-3 and D19x448 2e-3, 60 min, s0. ~$3.6 + ~$2.4.

### D34 (2026-11-05, 18:56-19:20): Phase 4 round 3 evaluated - Phase 4 DONE (Muon better; wide better at seed 0, deep worse)
- **Runs (7e1b431, launched 15:17-15:19):** E05 Muon 2e-3 60 min s1, s2 + 1e-3 s0 (giraffes573); E06 W6x768 2e-3 and D19x448 2e-3 60 min
  s0 (jinkyungjeon). All 5 terminals lost their connection (5 'failed' rows, annotated); recovered 18:56. All clean
  (committed, L4, 1 container, 1 attempt, cold compile, full 3,600 s, all four validation sets scored).
- **E05 horizon check (R1b):** 60 min s0: 2e-3 1.1686, 4e-3 1.1689, 1e-3 1.1716 -> the 15-min pick 2e-3 stands (no flip, so the AdamW
  fairness runs of R1c are not triggered).
- **E05 decision (R2): Muon BETTER.** 3 seeds 1.1686 / 1.1679 / 1.1684, mean **1.1683** vs AdamW N49 1.1945: **-0.0262** (threshold
  0.0091; ~2.9x). Robust: machine-speed adjusted -0.0246; every validation set -0.022 to -0.026 (AVG4 -0.0248); the 15-min gap was
  -0.059 and the gap along the hour shrinks then holds (D33). **Per step vs per time (R4a):** equal steps -0.030, equal time -0.028:
  Newton-Schulz costs Muon only ~0.002 BPB of its gain (it read 282M tokens vs 291M, -3 %). So Muon's advantage is per-step
  learning, not a speed effect. Seed sd 0.0003 (AdamW 0.0014): Muon runs are very consistent (descriptive; the declared pooled
  sigma60 is used). Clipped steps 1.1 % (all at the start). **Prediction (0.01-0.02 better) exceeded.**
- **E06 decision (R2, declared seed-0 rule, threshold 0.0132; no close-call seeds needed):**
  - **6x768 BETTER:** 1.1768, -0.0176 (speed-adjusted -0.0148, still > 0.0132; AVG4 -0.0154, every set better). It read 348M tokens vs
    291M (+16 %, MFU 0.257 vs 0.222). **Per step it ties** (equal steps -0.0016): the whole advantage is that wide layers run faster on
    the L4. **This is hardware-specific**: on a GPU where the 11 x 576 matmuls are already efficient, the gap would shrink.
  - **19x448 WORSE:** 1.2158, +0.0213 (speed-adjusted +0.0257; every set worse). Worse per step too (+0.015) AND slower (-12 %, worth
    ~+0.008).
  - Ordering width/depth 128 > 52 > 24 is monotone: under a time budget on this GPU, fewer wider layers win. **Prediction ("no shape
    clearly better") missed**; MobileLLM's deep-thin advantage does not appear at 49M under a time budget. Margin note: the wide
    verdict is one seed at 1.3x the threshold (declared rule); Phase 6 adds seeds if the wide shape enters the final recipe.
- **Consequences to settle BEFORE Phase 5/6 runs (not decided here):** (1) Phase 6's "best recipe": both Muon and 6x768 are REAL wins,
  but their combination is untested, Muon's LR was tuned at 11 x 576 only, and wide's gain is speed while Muon slows steps by ~4 %;
  a declared combination rule is needed. (2) Phase 5's model: vocabulary at the declared baseline (AdamW 11 x 576, compared with the
  existing 8k E02 runs, no new reference needed) or on the new recipe (needs an 8k reference of that recipe). (3) The
  `summarize.py E05` display issue (D32) no longer matters (3 seeds are in); fix it with the next code commit.
- **Phase 4 cost (ledger estimates):** 24 pilots + 7 sixty-minute runs ~ $11.

### D35 (2026-11-05, ~19:15-20:40): Phase 5 data built on giraffes573 and checked
- **Build:** `tools/vocab_app.py::build` (after the __pycache__ fix, D31 update), ~19:15; 4 CPU containers; build times 525-677 s.
- **Checks (20:21):** `fetch`: **8,192 REPRODUCTION CHECK: PASSED** (builder's 8k tokenizer = Phase 0's, hash
  3017aa1d6f9e: the recipe is identical at full scale). `verify`: **VOCAB DATA CHECK PASSED**, **baseline 8k files 13/13 unchanged**.
  Every split has the same documents and bytes as the baseline (checked inside the build).
- **Gap found in the check output and fixed:** the variant manifests recorded token-file paths as `/__modal/volumes/<volume id>/vocab/...`
  (the builder's path-safety check resolved the `/data` link and returned the resolved path). `verify` still hashed all 27 variant token
  files (they sorted above the visible part of the screenshot; PASSED is printed only if every expected file matched), but the paths
  are specific to giraffes573's volume. Fixed: local manifests `data/vocab/v*/manifest.json` normalised to `/data/vocab/...` (27 paths;
  `path_note` added; the Volume's own copies keep the resolved paths), and `_inside` now returns the path as given (test added: data root
  via a link; tools test passes). Phase 5 training reads `<data>/vocab/v<V>/...` by construction, not these recorded paths.
- **What the data shows (descriptive, FineWeb-Edu val bytes per token):** 4,096: 3.41; 8,192: 3.87; 16,384: 4.27; 32,768: 4.58. FineWeb-Edu
  train tokens: 2.78B / 2.45B / 2.22B / 2.08B. Each doubling of the vocabulary saves ~13 %, 10 %, 7 % of tokens (diminishing). Wikipedia
  compresses least at every size (2.94-4.07). Merges are nested (the 4,096 merges are the first 3,839 of 8,192, etc.): the tokenizers differ
  only in how many merges they keep. A 60-min run reads ~0.3B tokens, far below every train split (no repetition).
- **Next:** Phase 5 design (model, sizes, LR protocol, rules) before any Phase 5 run; needs the train.py data-variant field.

### D36 (2026-11-05, 20:30-21:15): review before Phase 5; Phase 5 (E07, vocabulary) designed and fixed before any E07 run
- **Context:** before Phase 5, a review of the plan and everything done so far, with attention to gaps,
  and a thorough Phase 5 design. Nothing has run since D35 (no launch is in progress), so code/ can change now.
- **Review of the plan and Phases 0-4: holds up; nothing to re-run.** Every comparison's rules were declared before its runs; failures,
  missed predictions and bugs are recorded. Gaps found (none changes a reported number):
  1. **The Phase 5 data build (D35) has no compute-ledger row** (CPU job, giraffes573, commit be6c519, ~19:15-19:27, 4 containers x
     525-677 s, 8 CPU / 32 GB). Row added (cost estimated ~$0.4 from Modal CPU/memory prices; the exact figure is on the usage page).
  2. **Ledger costs understate real cost:** the ledger's `cost_usd` is L4 GPU time only (~$0.83 per 60-min run); credits fall by ~$1.2
     per 60-min run (CPU, memory, compile, evaluation, image builds). The report's cost figures must come from the Modal usage pages,
     the ledger's from the GPU-hours. Stated in the E07 cost table; to be stated in the report.
  3. **Learning-curve subset is cut in TOKENS** (`curve_tokens` = 262,144, "about 1 MB at 3.87 bytes/token"): with another tokenizer it
     covers different text (4k ~0.89 MB, 32k ~1.20 MB), so curves of different vocabularies would not be comparable (only diagnostic,
     but the equal-bytes view of E07 needs them). Fixed for variants (below); the 8k runs keep their subset.
  4. **The Muon arm's embedding LR was never tuned** (E05 tuned `muon_lr`; embeddings/LayerNorm stayed at AdamW's 2e-3). This is the
     deciding reason for Phase 5's model choice (below) and a Phase 6 note: a vocabulary change under Muon would leave the only
     parameters it changes at an LR tuned for 8k.
  5. **Phase 6 test scoring must read the vocabulary variant** of a model trained on one: `rescore.py` now does (from the checkpoint's
     config); the future `code/evaluate_test.py` must reuse it.
  6. Known items unchanged: the original project plan (30 Oct) out of date -> a status section added; test lock still to
     build; Phase 6 combination rule still to declare; authors.md; invite alexxthiery; page limit 8 vs 10.
- **Phase 5 model: the declared baseline (AdamW 11 x 576), not the Phase 4 recipe.** Reasons: (1) **equal tuning effort:** the 8k
  reference had its single AdamW LR, which covers the embedding, tuned by the declared protocol (E01 N49, 4 LRs + close call); every
  vocabulary gets the same protocol on the same grid. Under Muon the embedding (the part the vocabulary changes) runs on a fixed AdamW
  LR tuned for 8k, so the comparison would be biased unless a new tuning protocol were invented, which the 8k Muon reference never had;
  (2) one factor changes (vocabulary), the Phase 4 changes stay out; (3) the 8k reference already exists (E02-N49, 3 seeds; E01 N49
  pilots), so no new 8k run is needed. Cost: Phase 6's combination of a vocabulary change with Muon/6x768 is untested; Phase 6's
  combination rule (to be declared) must handle it.
- **Sizes: 4,096, 16,384 and 32,768 (all three built), not the 2-size plan of D34/D35.** Why 16k is in the core set: (a) per-byte compute:
  FLOPs per token (6N + attention) 316M / 330M / 358M / 415M for 4k / 8k / 16k / 32k and FineWeb-Edu train bytes/token 3.43 / 3.88 / 4.27 /
  4.58 give text per FLOP 0.92 / 1.00 / 1.01 / 0.94 x 8k: 16k is the only size that reads at least as much text per hour; (b) Tao et al.
  (2024, Approach 1) put the compute-optimal vocabulary for our compute (9.6e16 FLOPs) at ~4.7k and for our non-embedding size (43.8M) at
  ~7.6k (their fit; compute-optimal data, FLOP- not time-limited); (c) 4k and 32k alone cannot tell whether 8k is a local optimum or a
  neighbour does better, i.e. whether the declared tokenizer choice was right. GPT-2 50,257 (DECLARATIONS sec. 4) is dropped: another
  build, a tokenizer trained on other text (confound), no credit. No edge extension beyond 4k-32k (budget): a best value at an edge is
  reported as such.
- **Rules (DECLARATIONS Amendments row, 5 Nov, "Phase 5 (E07)"), as E06:**
  - R0 data: every run hashes its variant's tokenizer and validation files (`data_check`); `summarize.py E07` compares them with the
    verified manifests (`data/vocab/v*/manifest.json`) and excludes a mismatch. Variant data exist only on giraffes573; another workspace
    needs `tools/vocab_app.py::build --vocab-sizes <V> --no-check` then `verify --vocab-sizes <V>` (VOCAB DATA CHECK PASSED) first.
    (Do NOT run `fetch` there: it would overwrite the committed local manifests.)
  - R1 LR per vocabulary: 5e-4, 1e-3, 2e-3, 4e-3 x 15 min, seed 0 (the grid N49 had); edge rule (<= 2 extensions per side; 2.5e-4 and
    8e-3 configs exist); close call (runner-up within 2.8 sigma15 = 0.0188) -> seeds 1, 2 of both, lower 3-seed mean wins. No 60-min
    horizon check (as E06; the baseline-size check was done at N26 in E01).
  - R2 per vocabulary: 60-min seed 0 at the pick vs E02-N49 3-seed mean 1.1945; real if |diff| > 2 sigma60 sqrt(4/3) = 0.0132; otherwise
    close call -> seeds 1, 2, 3-seed rule 1.6 sigma60 = 0.0091. **Priority** if credits run short: close calls with seed-0 diff < 0
    first (only those can change the recipe); one left unrun is reported as inconclusive (1 seed).
  - R3 replacement rule: speed > 10 % below that vocabulary's own pilots (the vocabulary changes speed by design), crash, warm
    compile, reused container, dirty code -> excluded until replaced (same seed, --fresh).
  - R4 best vocabulary: lowest mean among those BETTER than 8k; another BETTER one within 2 SE of it (2 sigma60 sqrt(1/na + 1/nb))
    -> optimal range; none BETTER -> 8k stays.
  - Decisions use FineWeb-Edu validation BPB only (declared objective). Reported, never decisive: D1 parameters (total / embedding /
    non-embedding); D2 tokens/s, bytes/s, MFU; D3 equal-bytes view (curves on the same text; time view minus bytes view = what the
    different text-per-second is worth); R4b machine-speed sensitivity; R4c AVG4 + per set; D6 loss per token vs per byte; D7 the 15-min
    comparison (R4e); D8 overlapping-window evaluation (stride 128) of the 60-min models and of E02-N49 s0-2: the declared windows reset
    the context every 512 TOKENS, i.e. every ~1.75 kB of text at 4k but ~2.34 kB at 32k, so part of a vocabulary gap can be an evaluation
    effect; D9 memory; D10 parabola vertex in log2 V; start loss vs ln V (sanity); LR sensitivity per vocabulary.
- **Code (bit-identical for every existing config, checked):**
  - `config.py`: `data_variant` ("" = baseline; "v<V>" must equal vocab_size; not v8192; fineweb_edu only, no mix).
  - `train.py`: data dir `<data>/vocab/v<V>/` for a variant; `byte_matched_prefix`: the variant's curve subset = the whole number of
    512-token windows whose bytes are closest to the baseline subset's (same documents in the same order, so same text to < 1 window,
    ~0.2 %); `data_check` and `curve_subset` written to the result (variant runs only).
  - `bpb.py`: `evaluate_bpb_sliding` (diagnostic only; equals `evaluate_bpb` exactly at stride = 512, tested). `rescore.py`: `stride`,
    scores a variant model on its variant's files, refuses a tokenizer/model vocabulary mismatch. `modal_app.py::rescore --stride N`.
  - `summarize.py E07` (reuses `_pick_lr`, `_final_vs_adamw`, `_slow_checks`, `_diagnostics` with the equal-steps view switched off, plus
    the E07 views). **D32 display issue fixed:** `summarize.py E05` no longer prints the single-seed rule for E05 (R2 always 3 seeds).
  - Tests 20/20 (new: variant data incl. no fallback to baseline files, byte-matched curve, hashes, rescore on the variant, baseline
    result has no new fields; sliding BPB = declared at stride T, every target once). Bit-identity: tiny AdamW, Muon and subset runs give
    identical weights, curves, BPB and result keys with old and new code (only the config gains `data_variant: ""`). `summarize.py
    E01-E06` output byte-identical to before. `summarize.py E07` exercised on simulated results (7 stages): clear pick, LR close call
    pending/decided, edge, seed-0 close call with priority list, BETTER/WORSE/inconclusive, 3-seed verdict, slow machine excluded, data
    mismatch excluded (pilot and final), optimum/edge note, parabola, sliding check.
  - 36 configs `configs/E07/V<V>_lr<x>_{15,60}min.yaml` (grid + 2.5e-4 / 8e-3 edges): differ from E02-N49 / E01 N49 pilots only in
    run_id, experiment, data_variant, vocab_size, lr (checked).
- **Risks checked:** GPU memory at 32k (logits 32 x 512 x 32,768; 8k peaked at 5.5 GB of 24) fine; host RAM: 4k train split 5.6 GB
  in RAM vs 4.9 GB at 8k (8k peak RSS 9.6 GB of the 16 GB request) fine; uint16 ids: max id 32,767 fine.
- **Prediction (before any run):** speed vs 8k: tokens/s 4k +2 to +5 %, 16k -5 to -9 %, 32k -12 to -20 %; text/s highest at 16k.
  BPB at 60 min vs 8k: 4k worse (+0.01 to +0.03), 16k within -0.015..+0.005 (most likely a close call), 32k -0.01..+0.02 (inconclusive or
  worse). Best: 8k or 16k. Overlapping windows help 4k most (more context resets per byte).
- **Budget (credits 20:26: giraffes573 $10.39, jinkyungjeon $10.69, joshuahanrc $5.66, umarmadbi ~$2; ~$0.25 per pilot, ~$1.2 per 60-min
  run):** round 1 = 12 pilots ~$3.0-3.6; typical total ~$9-11 (2 LR close calls, 3 seed-0 finals, 1 final close call); worst ~$15.
  giraffes573 first; never start a batch that would leave < ~$1.5; overflow on joshuahanrc (build that vocabulary there, ~$0.15).
  **jinkyungjeon is reserved for Phase 6** (only the D8 rescore of E02-N49, cents, runs there).

### D37 (2026-11-06, 00:30-00:45): E07 round 1 (LR pilots) evaluated; round 2 = 60-min seed 0 for all three vocabularies
- **Runs (commit afb895c = D36, launched 21:06, giraffes573; starts 22:13-~23:28 after a long Modal queue):** 12 pilots (4k / 16k / 32k x
  5e-4..4e-3, seed 0). Terminal kept its connection (no 'failed' rows). All clean: committed code, L4, 1 run per container, cold compile,
  900 s counted, no divergence; start loss ~ ln V + 0.1-0.15 at every size (sanity); curve subsets byte-matched (32k: 994,623 vs 995,694
  bytes, -0.11 %). **One interruption:** E07-V32768_lr5e-4_15min was lost after its 300-s checkpoint (Modal showed 2 'Failed' container
  attempts, 22:44 and 23:12) and resumed automatically at 23:28 from step 501 / 300.2 s (attempts 2; the declared resume path, budget
  carried over). Not an infrastructure failure under the replacement rule (it completed, speed within 6 % of its vocabulary's median);
  it is the worst LR of its grid, so it cannot affect the pick. Credits after round 1: giraffes573 $6.53 (round 1 cost ~$3.9).
- **R1 picks: 2e-3 for every vocabulary, all clear** (runner-up > 2.8 sigma15 = 0.0188 away: 4k 1e-3 +0.022, 16k 4e-3/1e-3 +0.023,
  32k 1e-3 +0.038), so no close-call or edge pilots. Same LR as 8k and every Phase 4 configuration.
- **15 min, seed 0 vs 8k 3-seed mean 1.3508 (R4e, descriptive; 2 SE = 0.0155):** 4k 1.3548 (+0.004), 16k 1.3426 (-0.008), 32k 1.3498
  (-0.001). All within noise; 16k lowest.
- **Speed (median of pilots):** tokens/s vs 8k: 4k +2.5 %, 16k -10.5 %, 32k -25.0 %. Text/s (x train bytes/token): 4k -9.5 %, 16k -1.4 %,
  32k -11.4 %. MFU 0.224 / 0.229 / 0.222 / 0.216 (4k/8k/16k/32k). **Prediction check:** 4k tokens/s held; 16k (-5..-9 %) and 32k (-12..-20 %)
  slower than predicted (the output layer + softmax cost more time than their FLOP share; MFU falls with V); "text/s highest at 16k"
  missed: 8k reads the most text per second (16k -1.4 %).
- **Display bug found (no number affected):** the D1/D2 table's "vs 8k" column prints +0.0 % for 4k (the 8k reference is computed after
  the 4k row). Values above computed by hand. Fix deferred until after the last E07 launch, so E07's code fingerprint stays the same
  (as D32).
- **Round 2 (as the rules print):** 60-min seed 0 at 2e-3 for 4k, 16k, 32k on giraffes573 (~$3.6 -> ~$2.9 left). Close calls (likely,
  given 15-min gaps < 0.01) cannot fit there afterwards (rule: keep >= ~$1.5), so the variant data is built on joshuahanrc NOW, in
  parallel (~$0.4, CPU), and verified before any close-call seed; saves a round.

### D38 (2026-11-06, 10:19-10:40): E07 round 2 (60-min seed 0) evaluated - 32k BETTER, 16k and 4k close calls; round 3 decided
- **Runs (afb895c, launched after D37, ~00:45; giraffes573; starts 01:36, 03:31, 03:55 after Modal queueing):** E07 V4096,
  V16384, V32768 at 2e-3, 60 min, seed 0. Launch terminal did not record them; recovered ~10:15 (3 rows; no 'failed' rows). All
  clean: committed, L4, 1 run per container, 1 attempt, cold compile (41-46 s), full 3,600 s, data_check = verified manifests, curve subset
  same text as 8k to <= 0.11 %, speed vs own pilots +2.6 / -2.9 / -2.1 %. joshuahanrc: variant data built and `verify` PASSED (~10:15).
  Credits: giraffes573 $2.98.
- **R2 (seed 0 vs 8k E02-N49 mean 1.1945; threshold 0.0132):** 4k 1.2032 (+0.0087, CLOSE CALL), 16k 1.1841 (-0.0104, CLOSE CALL),
  **32k 1.1774 (-0.0171, BETTER**; 1.3x the threshold, as the 6x768 verdict; speed-adjusted -0.0161, still beyond).
  **BPB falls monotonically with vocabulary size: 1.2032 / 1.1945 / 1.1841 / 1.1774 for 4k / 8k / 16k / 32k.**
- **Robustness (reported, not decisive):** AVG4 4k +0.011, 16k -0.008, 32k -0.011; every set has the same sign as FineWeb-Edu, but the gains
  are largest on FineWeb-Edu (32k: -0.017 FWE vs -0.007..-0.010 elsewhere); 4k is much worse on DCLM (+0.025). **D3 equal bytes:** 32k reads
  11 % less text in 60 min (219M tokens x 4.58 B vs 291M x 3.88 B) yet is -0.025 better at equal text; its slower text rate costs it ~0.010.
  16k -0.009 at equal bytes; 4k +0.007 (it reads 4.6 % less text despite +8 % tokens/s). So the larger-vocabulary gain is per byte of text
  seen, not speed; speed works against it. Clipping 0.2-0.5 %, peak GPU 5.1 / 6.4 / 8.2 GB, host RSS <= 10.2 GB.
- **Prediction check (D36):** 4k worse - direction held (close call); 16k close call - held; **32k "-0.01..+0.02, inconclusive or worse" -
  MISSED; "best 8k or 16k" - MISSED** (best is the largest size tried). Tao et al.'s compute-optimal ~5-8k does not hold here; candidate
  reasons to examine in the report: (a) the token-based context window holds 2.34 kB of text at 32k vs 1.98 kB at 8k (D8 tests the
  evaluation part); (b) tied embeddings add capacity (63.0M vs 48.8M total; E02's flat optimum 36-65M); (c) a larger vocabulary makes each
  prediction cover more text, so the 11-layer model spends its depth on fewer, richer positions.
- **Gap found and acted on: is the 32k gain an evaluation artifact?** The declared windows reset the context every 512 tokens, i.e. every
  ~1.75 / 1.98 / 2.18 / 2.34 kB at 4k / 8k / 16k / 32k. The D8 overlapping-window rescore (declared in D36) is therefore run NOW, not at
  the end: E07 60-min models (giraffes573) and E02-N49 s0-2 (jinkyungjeon), FineWeb-Edu, stride 128. Cents. Never decisive.
- **Round 3 (D36 priority rule):** (1) **16k seeds 1, 2 (priority: seed-0 diff < 0) on joshuahanrc** (giraffes573's $2.98 cannot hold two
  60-min runs with the ~$1.5 margin). (2) 4k seeds 1, 2: lower priority (can only end inconclusive or worse); run only if credits allow
  after Phase 6 is costed; otherwise reported as a 1-seed close call (+0.0087). (3) 32k needs no more seeds under the declared rule; if 32k
  enters the Phase 6 recipe, Phase 6 runs its seeds (as for 6x768).
- **Edge:** the best vocabulary is at the top of the grid. D36 declared no extension; a 65,536 vocabulary (the uint16 limit) would be a new
  follow-up needing its own declaration, build and protocol (~$2.5-3.5). Not run now: Phase 6 now has three untested winners to combine
  (Muon, 6x768, 32k) and needs the credits. Reported as a limitation ("optimum at or beyond 32k for this model and budget").

### D39 (2026-11-06, 10:20-10:35): D8 evaluation-context check - about a third of the vocabulary gap is the scoring window; the ordering holds
- **Runs (~10:20-10:27, commit afb895c):** `rescore --sets fineweb_edu --stride 128` on giraffes573 (E07 60-min s0 x 3) and
  jinkyungjeon (E02-N49 s0-2). Every declared score reproduced exactly (diff 0.0e+00). Files results/rescore/*__slide128.json.
- **Result (FineWeb-Edu val; sliding = overlapping windows, each scored token has >= 384 tokens of context):**
  | vocab | declared | sliding | gain | gap to 8k declared | gap to 8k sliding |
  |---|---|---|---|---|---|
  | 4k | 1.2032 | 1.1657 | -0.0374 | +0.0087 | +0.0055 |
  | 8k (3 seeds) | 1.1945 | 1.1602 | -0.0342 | 0 | 0 |
  | 16k | 1.1841 | 1.1532 | -0.0309 | -0.0104 | -0.0070 |
  | 32k | 1.1774 | 1.1489 | -0.0285 | -0.0171 | -0.0113 |
  8k sliding seed sd 0.0015 (declared 0.0014): the overlapping evaluation is no noisier.
- **Interpretation:** the gain from overlapping windows shrinks steadily with vocabulary size (as predicted: small vocabularies reset
  the context more often per byte). It removes about one third of each gap (4k 37 %, 16k 33 %, 32k 34 %), so ~1/3 of the vocabulary
  effect under the declared protocol is the evaluation window holding more text, and ~2/3 is a better model of the text. The ordering
  4k > 8k > 16k > 32k (BPB) is unchanged. Note: under the sliding view the 32k gap (-0.0113) is below the single-seed threshold
  (0.0132); the declared verdict (BETTER, D38) stands because the decision metric was fixed in advance, and the report states both.
- **Consequence for the report and Phase 6:** a vocabulary advantage measured with token-based windows overstates the modelling gain by
  ~1/3. If 32k enters the Phase 6 recipe, the test score is reported under the declared protocol with this caveat (and the sliding score
  beside it).

### D40 (2026-11-06, 11:44-12:00): E07 round 3 - 16k BETTER with 3 seeds; optimal range {16k, 32k}; Phase 5 decided except the 4k close call
- **Runs (afb895c; launched ~10:23 on joshuahanrc, whose variant data verified ~10:15):** E07-V16384_lr2e-3_60min seeds 1, 2. Reached the
  laptop through the launch terminal (no 'failed' rows). Clean: committed, L4, 1 run per container, 1 attempt, cold compile (32-54 s),
  full 3,600 s; data_check equals the verified manifest (same token_bytes hash as giraffes573's run); speed vs 16k pilots +3.9 / +0.7 %.
- **R2 16k (3 seeds): 1.1841 / 1.1734 / 1.1769, mean 1.1781 (sd 0.0054) vs 8k 1.1945: -0.0163 -> BETTER** (threshold 1.6 sigma60 =
  0.0091; 1.8x). Robust: machine-speed adjusted -0.0116 (still beyond; seeds 1, 2 ran on faster machines, +3.9 / +0.7 % vs -2.9 % for
  seed 0, and the fastest run is the lowest: speed explains much of the seed spread, as in E02); AVG4 -0.0143 (beyond 1.6 sigma_avg4 =
  0.0088); every set better (-0.012 to -0.016). Seed 0 alone (-0.0104) was the unluckiest of the three.
- **R4 best vocabulary:** BETTER than 8k: 16k (3 seeds, 1.1781) and 32k (1 seed, 1.1774). Difference 0.0007, far below 2 SE (0.0132)
  -> **optimal range {16k, 32k}**; 32k is at the grid edge (limitation, D38). The 8k declared baseline is ~0.016 BPB from the best
  vocabulary at this model size and budget; the decision made on 30 Oct (keep compute in the transformer) was too conservative.
- **D8 (D39) applied to 16k:** seed 0's sliding gap was -0.0070 (about 2/3 of its declared gap); seeds 1, 2 not rescored (could be done
  on joshuahanrc for cents; not needed for any decision).
- **Prediction check (D36):** 16k "-0.015..+0.005, most likely a close call" - held at seed 0 (-0.0104 close call); the 3-seed mean
  (-0.0163) is just past the predicted range.
- **Still open: 4k close call** (seed 0 +0.0087; can only end inconclusive or worse). Credits are now ~$2.9 (giraffes573) and ~$2.8
  (joshuahanrc, estimate), so two more 60-min runs would push both workspaces close to the ~$1.5 margin that Phase 6's test scoring there
  needs. **Decision: deferred to the Phase 6 costing**; if not run, reported as a single-seed close call ("4k not better; probably worse").
- **Phase 5 conclusion for Phase 6:** a vocabulary in {16k, 32k} is a REAL improvement at the baseline model; the choice between them and
  how to combine with Muon (whose embedding stays on AdamW) and 6x768 belongs to the Phase 6 combination rule, declared before any run.
  Considerations to carry over: 16k has 3 seeds and costs less per token (-10 % vs -25 % tokens/s); 32k's gain under overlapping windows is
  smaller relative to its declared gap only as much as 16k's (~1/3 each); with 6x768 the embedding grows (V x 768).
- Deferred code fix (D37 display bug in the D1/D2 table) stays deferred until the 4k decision is made (no code change between E07 runs).

### D41 (2026-11-06, 11:50-12:10): review of the plan and Phases 0-5 before Phase 6
- **Context:** before Phase 6, a review of the plan and everything done so far, looking
  for anything done wrong, before Phase 6 is designed. Nothing is running (no launch since D40), so code/ can change.
- **Checked (re-read or re-run, not just the records):** `bpb.py` (EOT excluded from loss and bytes; float64 sums; every target
  once; sliding = declared at stride T), `rescore.py` (variant files, vocabulary check), the Muon split in `train.py`
  (44 matrices on Muon, tied embedding / position / LayerNorm on AdamW, one schedule via lr_scale, clipping over all
  parameters, PyTorch's Muon), `config.py` validation, the E05-E07 configs (differ from their parents only in the intended
  fields), the noise thresholds (1 vs 3: 2 sigma sqrt(4/3) = 0.0132; 3 vs 3: 1.6 sigma = 0.0091), the contamination file
  (covers val AND test of all four sets), every 60-min run's workspace (from its own result meta). `code/tests` 20/20
  (torch 2.15 CPU); `summarize.py E01-E07` reproduce the recorded numbers.
- **Verdict: nothing in Phases 0-5 is wrong in a way that changes a reported number; nothing to re-run.** Every
  comparison's rules were declared before its runs; failures, missed predictions and bugs are recorded.
- **Small items found (none changes a decision):**
  1. `summarize.py E07` D1/D2 "vs 8k" column for 4k (D37 bug, deferred while E07 could still launch): fixed now (D42); the
     only changed output line is 4k's, now -9.5 % (the value D37 computed by hand). E01-E06 output byte-identical.
  2. The status notes still listed D37-D40 as uncommitted: they are committed (5fe3dc3). The repo is **7
     commits ahead of GitHub** (last push = a1dd2b9, D27): push, to keep an off-laptop copy and before the professor
     needs access.
  3. **Test-set contamination:** results/contamination.json shows FineWeb-Edu test 3.3 %, FineWeb 1.2 %, DCLM 1.2 % of
     documents contaminated (>= 50 % 13-gram overlap with some training corpus), Wikipedia 0.2 %. The declared rule (sec. 3)
     removes them before final evaluation; validation got this in E03 R3 but no rule existed yet for test -> D42: every
     test score is also given without them (same documents for all models).
  4. Known open items carried into D42: test lock not built; combination rule not declared; Muon's embedding LR never
     tuned (D36 gap 4); 6x768's single-seed verdict (D34 margin note); 4k close call (D40); N26 s0 lacks 3 val sets;
     original project plan outdated; authors.md; invite alexxthiery; page limit 8 vs 10.
  5. Multiple comparisons: about a dozen REAL/WORSE calls were made at ~2 SE each; two of the "REAL" wins (6x768, 32k) are
     single seeds at 1.3x the threshold. Not a rule change: the report states it, and Phase 6's 3-vs-3 runs re-test both
     (width inside the recipe; 16k, not 32k, is carried forward).
- **Plan check:** the plan's Phase 6 ("baseline vs best combined recipe, 3 seeds each, test once") still fits; the budget is
  now ~$18 (not ~$75 reserve), so Phase 6 must be designed to the per-workspace credits (D42).

### D42 (2026-11-06, 12:10-12:45): Phase 6 (E08) designed and fixed before any E08 run; test lock built
- **Full design:** `docs/experiments/E08_final_recipe.md`; DECLARATIONS Amendments row "Phase 6 (E08)". Summary:
- **Two recipes, 3 seeds each:** RA = Muon + 6x768 + 16k (all REAL winners); RB = Muon + 11x576 + 16k. Why both: the wide
  shape's gain is purely speed (E06: equal per step, +16 % tokens), and both other changes eat speed more at width 768
  (Newton-Schulz +29 %, output layer +33 %), so width may not pay inside the recipe; RB measures that 3 vs 3 (replacing the
  extra E06 seeds D34 promised), is the fallback recipe, and with the existing runs gives a clean Muon x vocabulary
  interaction (partial factorial: B, M, W, V, MV, MWV; M+W and W+V not run, budget).
- **Considered and rejected:** (a) RA only (cheapest, ~$5.4): cannot tell whether width helps in combination, no attribution;
  (b) full 2x2x2 factorial (4 new cells with tuning, ~$15+): does not fit the credits; (c) skipping LR tuning for the
  recipes because every configuration so far picked 2e-3: breaks the equal-tuning-effort principle on the headline
  comparison; (d) 32k in the recipe: tie with 16k, declared tie-break for 16k (3 seeds vs 1, -10.5 % vs -25 % tokens/s, the
  gap grows at width 768, smaller embedding under the untuned embedding LR, same 1/3 window caveat); (e) a fresh baseline
  seed as a drift control (~$1.2): training code unchanged since afb895c, speed checks and R4b cover machine drift, and
  every L4 so far ran power-capped at 72 W; (f) tuning the embedding LR as a second decisive knob: more tuning effort than
  the baseline got -> made a diagnostic (R1e); (g) a horizon check for the recipes: E05 checked Muon (held), E06/E07 had
  none.
- **Final recipe rule:** RA if lower than RB beyond 1.6 sigma60 (0.0091), RB if RB lower beyond it, RB on a tie (fewer
  changes; not tied to L4-specific speed); RA if RB has < 3 usable seeds.
- **4k close call (D40): dropped.** Seeds 1, 2 (~$2.4, needing the 4k data on a workspace with credit) can only end
  "inconclusive" or "worse" and change no recipe; the credits go to RB's 3 seeds instead. Reported as a single-seed close
  call (+0.0087, "4k not better; probably worse").
- **Test lock built (DECLARATIONS sec. 5):** `code/evaluate_test.py` (refuses unlisted runs or a non-frozen list; scores the
  TEST split of the model's own vocabulary variant via `rescore.rescore(split="test")`; same documents minus the declared
  contaminated ones; overlapping windows (stride 128) on FineWeb-Edu for core models; 3 text samples from fixed prompts for
  marked runs), `modal_app.py::test_one` / `test_eval` (refuses uncommitted code or frozen list and any run already scored on
  the laptop or the Volume; the container re-checks its frozen list against the laptop's; one container per workspace, one
  ledger row per workspace; `--tiers`, `--fetch-only`), `configs/frozen_configs.yaml` drafted (status `draft`; 51 completed
  runs in tiers core 13 / data 9 / extra 29; every workspace matches the run's own result meta; E08 finals added at
  freeze time).
- **Code changes (training path untouched: train.py, model.py, bpb.py, config.py identical to afb895c):** `rescore.py`
  (`split` argument, default "val" = old behaviour; `data_dir_for` helper; result gains a `split` key); `summarize.py`
  (`E08`, `TEST`; E07 display fix); `modal_app.py` (`test_one`, `test_eval`); `code/tests/test_pipeline.py` (22 tests: new
  "combined recipe: Muon + variant + shape, resume, rescore" and "test lock"; the config-loading test skips the frozen
  list). Checks: 22/22 pass (torch 2.15 CPU); `summarize.py E01-E06` byte-identical, E07 only the fixed cell; `E08` simulated
  through every branch (waiting -> launch lines, clear pick, close call pending/decided, edge, R1e, slow machine excluded,
  data mismatch excluded, 1-2 of 3 seeds waiting, 3-seed decisions, RA/RB tie -> RB, attribution, D8); `TEST` simulated.
- **48 configs `configs/E08/`:** per recipe muon_lr {2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2} x {15, 60} min, and RA
  R1e configs at every muon_lr. Checked: RB differs from E05's Muon config only in vocab/data_variant/run_id, from E07 16k
  only in optimizer/muon_lr/run_id; RA differs from E06 6x768 only in optimizer/muon_lr/vocab/data_variant/run_id.
  Parameters: RA 55.5M (embedding 12.6M, non-embedding 42.5M), RB 53.5M (9.4M, 43.8M); FLOPs/token 359M / 358M.
- **Prediction (before any run):** in the E08 record (both pick 2e-3; RA tokens/s -2..-8 %, RB -12..-17 %; RB 1.148-1.158,
  RA 1.138-1.152, sub-additive by 0.003-0.010; RA vs RB most likely inconclusive -> RB final; both BETTER than baseline and
  than Muon alone; R1e within noise; test ordering = val, Spearman > 0.9).
- **Order and placement (credits to be confirmed):** round 0: v16384 build + verify on jinkyungjeon (and
  umarmadbi, optional), N26 s0 rescore (umarmadbi), D8 rescore of 16k s1, s2 (joshuahanrc); round 1: RA pilots giraffes573,
  RB pilots joshuahanrc; round 2: close calls (jinkyungjeon; a second one on umarmadbi); round 3: six finals on
  jinkyungjeon + R1e on giraffes573; round 4: D8 rescore of the finals, decisions, freeze + commit; round 5: test_eval per
  workspace. Expected ~$13-14 of ~$18.

### D43 (2026-11-06, 13:46-14:05): E08 round 1 (LR pilots) evaluated - both recipes are LR close calls; round 0 extras done
- **Round 0 (~12:30-13:40, commit 776406f = D41-D42, pushed):** jinkyungjeon v16384 build + `verify`: VOCAB DATA CHECK
  PASSED, baseline 8k files 13/13 unchanged. umarmadbi: N26 s0 (E01-N26_lr4e-3_60min) rescored on the 3 missing val sets: FineWeb
  1.3969, DCLM 1.5525, Wikipedia 1.4742 (lowest of the 5 N26 seeds on every set, as on FineWeb-Edu: consistent; E02 AVG4 now
  complete). joshuahanrc: D8 rescore (stride 128) of E07 16k s1, s2: declared scores reproduced exactly (diff 0.0).
- **D8 with 3 seeds of 16k (corrects D39/D40's "~1/3" for 16k):** 16k declared 1.1841 / 1.1734 / 1.1769, sliding 1.1532 / 1.1425 /
  1.1462. Gap to 8k (3 seeds each): declared -0.0163, **sliding -0.0129** (still beyond the 3-vs-3 threshold 0.0091). The window
  effect is a near-constant ~0.0034 BPB per model (sliding gain 16k 0.0308 vs 8k 0.0342, every 16k seed 0.0307-0.0309), i.e.
  ~21 % of the 3-seed 16k gain, not 1/3: D39's ratio used seed 0, whose gap was the smallest. 32k (1 seed): 0.0057 of 0.0171
  (~1/3) stands. Report: give the absolute window effect (16k ~0.003, 32k ~0.006) and the sliding gaps.
- **Round 1 runs (776406f, launched ~12:29-12:30; giraffes573 RA, joshuahanrc RB):** 8 pilots, all clean: committed, L4, 1 attempt,
  cold compile (30-44 s), 900 s, no divergence, data_check = verified manifest. Credits after (13:46): giraffes573 $1.63,
  joshuahanrc $1.61, jinkyungjeon $10.49, umarmadbi $2.35.
- **RA (Muon + 6x768 + 16k), 15 min s0:** muon_lr 1e-3 1.2627, 2e-3 1.2645, **4e-3 1.2625**, 8e-3 1.3896. **CLOSE CALL 4e-3 vs 1e-3**
  (gap 0.0002; 2e-3 is also within 0.002: a flat basin over a factor 4). Declared rule (as E05): seeds 1, 2 of the best two
  (4e-3, 1e-3) only; 2e-3 is not rerun. 8e-3: no spike or divergence; validation progress stalls from step ~500 to ~1,100
  (1.671 -> 1.585), i.e. too high an LR for the wide shape (E05's Muon at 11x576 lost only 0.025 at 8e-3).
- **RB (Muon + 11x576 + 16k), 15 min s0:** 1e-3 1.2923, **2e-3 1.2711**, 4e-3 1.2813, 8e-3 1.3073 -> **CLOSE CALL 2e-3 vs 4e-3** (gap
  0.0101), the same pair as E05.
- **Clipping:** RA 1e-3 5.4 %, RB 1e-3 10.2 % of steps, all in the first ~165 steps (the start-up gradient norm ~9 takes longer to fall
  below the clip at a lower Muon LR; 0 % after). Not an instability (as E05, D32); recorded because RA 1e-3 is a contender.
- **Speed (D2, median of pilots):** RA tokens/s **+0.1 %** vs the baseline (text/s +10.4 %), RB -13.9 % (text/s -5.1 %). RA vs RB
  +16.3 %, the same as 6x768 vs 11x576 under AdamW (+16 %); Muon + 16k cost the wide shape -14.0 % (vs AdamW 6x768 8k) and the
  narrow one -13.9 % (vs the baseline): **the costs are proportional, width's speed advantage survives in the recipe.**
  Prediction check: RB (-12..-17 %) held; RA (-2..-8 %) missed (faster): the D42 reasoning that NS and the output layer would erode
  width's advantage more at 768 is not borne out on the L4. RA's pilots spread 82.4k-89.1k tok/s (1e-3 the fastest, +6 % over the
  median; within the 10 % rule); the close-call seeds also average out this machine luck.
- **15 min (descriptive, 1 seed):** RA ~1.2626 vs RB 1.2711 (-0.009); vs AdamW baseline 1.3508: RA -0.088, RB -0.080; vs Muon 8k
  (1.2914, 3 seeds): RA -0.029, RB -0.020. 15-min gaps overstate 60-min ones (E05: x2.3).
- **Round 2 (as the rules print):** RA lr 1e-3, 4e-3 and RB lr 2e-3, 4e-3, 15 min, seeds 1, 2 (8 pilots, ~$2.4) on **jinkyungjeon**
  (16k verified; leaves ~$8.1). In parallel: v16384 build + verify on **umarmadbi** (~$0.15), so that one 60-min final can run
  there in round 3 (jinkyungjeon then runs 5 finals, ~$6.0 -> ~$2.1 left for test scoring and reserve). R1e waits for RA's pick
  (round 3, giraffes573).

### D44 (2026-11-06, 14:45-14:55): E08 round 2 (close-call pilots) evaluated - picks RA muon_lr 4e-3, RB 2e-3; round 3 = finals
- **Runs (776406f, jinkyungjeon, launched ~13:57):** RA 1e-3 / 4e-3 and RB 2e-3 / 4e-3, 15 min, seeds 1, 2. All clean (committed,
  L4, 1 attempt, 900 s, no divergence, data_check = manifest), reached the laptop. umarmadbi: v16384 built, `verify` PASSED
  (baseline 13/13 unchanged).
- **R1 RA: muon_lr 4e-3** (3-seed means 4e-3 1.2627 vs 1e-3 1.2638; seeds 1.2625/1.2661/1.2597 vs 1.2627/1.2619/1.2668). The margin
  (0.0011) is far inside the noise: RA's basin is flat over 1e-3-4e-3 (2e-3 s0 1.2645 lies between). The declared rule (lower
  3-seed mean) picks 4e-3 and is applied as declared. 1e-3's seeds ran on faster machines (88.6-89.1k vs 81.6-85.7k tok/s) and
  still lost, so machine luck did not decide it. **Risk noted, not acted on:** the optimal LR tends to fall with run length
  (Bjorck et al. 2024) and 8e-3 already stalls the wide shape, so the higher of two tied values could do slightly worse at 60 min.
  Evidence against a material effect: E05's 60-min horizon check found Muon 4e-3 = 2e-3 (1.1689 vs 1.1686) at 11x576. Changing
  the pick after seeing it would break the declared protocol; no extra horizon runs (budget; E06/E07/E08 have none). The report
  states it; RA's 60-min learning curves are checked for late stalling.
- **R1 RB: muon_lr 2e-3** (2e-3 1.2735 vs 4e-3 1.2841, -0.0106), the same pick as E05's Muon at 11x576.
- **15 min, 3 seeds each (descriptive):** RA 1.2627 vs RB 1.2735 (-0.0108); vs AdamW baseline 1.3508: RA -0.088, RB -0.077.
- **Speed (all pilots):** RA tokens/s +1.0 % vs baseline (text/s +11.3 %), RB -13.7 % (text/s -4.8 %).
- **Round 3 (as the rules print):** RA 4e-3 60 min s0, s1, s2 + RB 2e-3 60 min s0, s1 on jinkyungjeon (~$6.0 of ~$8.1); RB 2e-3 60 min
  s2 on umarmadbi (~$1.2 of ~$2.2); R1e on giraffes573: RA muon_lr 4e-3 with the AdamW part at 1e-3 and 4e-3, 15 min, s0 (~$0.6 of
  $1.63). Credits are then ~$2.1 / ~$1.0 / ~$1.0 / $1.61 (jinkyungjeon / umarmadbi / giraffes573 / joshuahanrc) for test scoring.

### D45 (2026-11-06, 17:40-18:05): E08 round 3 (finals) evaluated - both recipes BETTER; RA vs RB not separated -> RB is the declared final recipe; RA the best measured; test list frozen
- **Runs (776406f, launched ~14:49-14:50):** RA 4e-3 60 min s0-2 and RB 2e-3 s0, s1 (jinkyungjeon), RB 2e-3 s2 (umarmadbi), R1e 2
  pilots (giraffes573). All clean: committed, L4, 1 attempt, cold compile (30-45 s), full 3,600 s (900 s for R1e), data_check =
  verified manifests, speed vs own pilots -3.2..+3.5 % (no slow machine), all four val sets scored. Reached the laptop (no
  'failed' rows).
- **Finals (FineWeb-Edu val):** RA 1.1453 / 1.1428 / 1.1449, mean **1.1444** (sd 0.0013); RB 1.1537 / 1.1554 / 1.1502, mean
  **1.1531** (sd 0.0027).
- **R3 decisions (declared: 3 vs 3, real if > 1.6 sigma60 = 0.0091):**
  - RA vs baseline 1.1945: **-0.0501 BETTER** (speed-adjusted -0.0475; AVG4 -0.0510; every set -0.046..-0.057).
  - RB vs baseline: **-0.0414 BETTER** (speed-adjusted -0.0376; AVG4 -0.0406; every set -0.038..-0.044).
  - RA vs Muon alone 1.1683: -0.0239 BETTER; RB vs Muon alone: -0.0152 BETTER. Combining beats the best single change.
  - **RA vs RB: -0.0087, threshold 0.0091 -> not separated -> FINAL RECIPE = RB** by the declared tie-break.
- **The honest reading of RA vs RB (reported beside the decision, never replacing it):** the declared threshold uses sigma60 =
  0.0057, the seed sd of AdamW runs (84 % of it machine speed, D19). The recipes vary 2-4x less (RA 0.0013, RB 0.0027); with the
  arms' own sds 2 SE = 0.0034, and every RA seed is below every RB seed. So under the recipes' own noise RA is better by ~0.009;
  under the declared, conservative rule it is a near-miss (0.96 of the threshold). Not changed after seeing the result (amendment
  policy (2): a comparison's rules are never changed after its first run); not resolved with extra seeds either (unplanned
  extra seeds after a near-miss = optional stopping, and ~$2.4 is not available beside test scoring). Report wording: "RB is the
  declared final recipe; RA (all three changes) has the lowest validation BPB measured, 1.1444, 0.0087 below RB: inconclusive
  under the pre-declared noise scale, beyond noise under the recipes' own seed variance." Both recipes are frozen and scored on
  test, so the report gives both test results.
- **No late stall at RA's 4e-3 (D44 risk):** learning-curve gap RA - RB (same text subset): -0.019 at 10 %, ~0 from 30 % to 70 %,
  then -0.004 / -0.006 / -0.009 at 80 / 90 / 100 %: RA pulls ahead during the final LR decay. The higher tied LR did not hurt.
- **R1e (diagnostic):** RA at muon_lr 4e-3 with the AdamW part at 1e-3 1.2702, 4e-3 1.2659 vs 2e-3 1.2627 (3 seeds): 2e-3 is the
  best tried -> the inherited embedding LR is not leaving gain (D36 gap 4 closed for this recipe).
- **D-attribution (summarize E08):** 16k given Muon -0.0152 (alone -0.0163); Muon given 16k -0.0251 (alone -0.0262); **Muon x 16k
  interaction +0.0011 (2 SE 0.0132): additive** - the two gains act independently. 6x768 given Muon + 16k -0.0087 (alone -0.0176,
  1 seed): width keeps all its speed advantage (+16 % tokens/s, D43) but each extra token is worth less once Muon learns faster
  per step; non-additivity of all three +0.0100 (2 SE 0.0208, wide because W has 1 seed): mildly sub-additive, all of it from width.
- **Other diagnostics:** RA reads 299M tokens (+3 % vs baseline 291M), 1.279 GB of text (+13.6 %); equal-bytes view RA - 8k -0.038
  (so ~0.013 of RA's gain is reading more text per hour, ~0.038 is learning more per byte). RB reads 257M tokens, 1.099 GB (-2.4 %),
  equal-bytes -0.043: all of RB's gain is per byte. Clipping 0.8 % / 1.25 % (start only). Peak GPU 5.2 / 6.2 GB.
- **Prediction check (D42):** RB 1.148-1.158: held (1.1531). RA 1.138-1.152: held (1.1444). RA - RB -0.004..-0.012, "most likely
  inconclusive -> RB final": held (-0.0087). Sub-additive by 0.003-0.010: held (+0.0100 all three; Muon x 16k additive). Both
  BETTER than baseline and Muon alone: held. R1e within noise: held. RA speed (-2..-8 %): missed (+1 %, D43).
- **Test list frozen** (`configs/frozen_configs.yaml`, status frozen, 57 runs): the six E08 finals added under core (RA s0, RB s0
  with text samples). Every workspace matches the run's own result meta; `check_lock` passes. Must be committed before
  `test_eval`. Next (all four workspaces, in parallel): D8 overlapping-window rescore of the six finals (val, diagnostic) and the
  one-time test scoring.

### D46 (2026-11-06, 18:06-18:20): first test_eval attempt cancelled by a dropped connection (our bug: .remote()); fixed; D8 of the E08 finals
- **What happened (~17:50-17:55, commit 3ba1d90):** `modal run --detach modal_app.py::test_eval` on all four workspaces. All four
  stopped at about the same time with "Received a cancellation signal while processing input" in the container log and a
  traceback at `test_one.remote(...)` on the laptop: the laptop's connection dropped (as on most launches) and Modal **cancels a
  `.remote()` call when its client disconnects, even with --detach**. `launch`, `rescore` and `rebuild_data` use `spawn` exactly to
  survive this (D16); `test_eval` (D42) did not. My bug.
- **Effect on the test lock: none.** `test_one` saves each model's result on the workspace's runs Volume (and commits) before
  starting the next one, so every model scored before the cancel is kept and can never be re-scored (the container refuses it
  as "ALREADY SCORED"); nothing reached the laptop (results/test/ empty, no TEST ledger row). On giraffes573 the log shows the
  three E05 models finished and E07-V16384 s0 had started. The model in progress when cancelled was not saved, so it is scored
  (once) on the retry: no score has been seen, so this is not a second scoring.
- **Fix (modal_app.py only; scoring code unchanged):** `test_eval` now `spawn`s `test_one` and waits with `.get()`; if the connection
  drops it says to fetch later with `--fetch-only` (the spawned call keeps running). `test_one` catches any exception per model
  (records it, does not save, continues), so one failing model cannot stop the rest. Must be committed before the retry
  (test_eval refuses uncommitted code).
- **Retry procedure:** per workspace `modal run --detach modal_app.py::test_eval` (already-scored models come back "ALREADY
  SCORED" and are skipped), then `modal run modal_app.py::test_eval --fetch-only` to bring every saved result to results/test/ and
  write the ledger row.
- **D8 overlapping-window rescore of the E08 finals (val, stride 128; every declared score reproduced exactly):** RA 1.1146 / 1.1121 /
  1.1141 (mean 1.1136), RB 1.1222 / 1.1240 / 1.1187 (mean 1.1216); 8k baseline sliding mean 1.1602. Gaps to the baseline: RA declared
  -0.0501 -> sliding -0.0466; RB -0.0414 -> -0.0386; RA - RB -0.0087 -> -0.0080. The window effect is ~0.003 (as for 16k alone, D43):
  ~93 % of each recipe's gain is a better model of the text, not the scoring window.

### D47 (2026-11-06, 19:10-19:30): one-time test scoring done (57 frozen models) - every validation conclusion holds on test; Phase 6 DONE
- **Runs:** `test_eval` per workspace (ledger rows TEST-<workspace>): umarmadbi 6 models (scored 18:10-18:12 SGT from commit 3ba1d90
  with the pre-D46 `.remote()` path, which this time was not cut off: clean commit, scoring code identical to 7102de7);
  giraffes573 13 (the three E05 models saved by the cancelled first attempt at 17:47-17:52, the rest on the retry), joshuahanrc
  22, jinkyungjeon 16 (retry from 7102de7, then `--fetch-only`). Checks: all 57 listed models present, none extra; every result
  carries the same frozen-list commit (3ba1d90) and hash (2f418d33); split = test; all on NVIDIA L4; every entry got the sets,
  contamination-free score, overlapping-window score and samples its tier asked for. No model was scored twice. Scoring GPU time
  ~1.0 h in total (2,052 / 1,080 / 249 / 208 s).
- **Val-test agreement:** test = val + 0.0053 on average, the same offset (+0.0046..+0.0062) for every group; Spearman rank
  correlation over 57 models 1.000. **Removing the contaminated test documents changes no difference by more than 0.0008.**
- **Headline (FineWeb-Edu test, 3 vs 3, declared noise scale 0.0091):**
  | arm | test BPB | vs baseline | without contaminated docs | overlapping windows |
  |---|---|---|---|---|
  | baseline AdamW 11x576 8k | 1.1994 +- 0.0015 | - | - | - |
  | RB Muon + 11x576 + 16k (declared final) | 1.1583 +- 0.0025 | **-0.0411 BETTER** | -0.0404 | -0.0386 |
  | RA Muon + 6x768 + 16k (best measured) | 1.1494 +- 0.0011 | **-0.0500 BETTER** | -0.0492 | -0.0469 |
  RA - RB on test -0.0089 (w/o contaminated -0.0088, sliding -0.0082): the same near-miss as on val (threshold 0.0091; with the
  arms' own sds clearly beyond noise). Reported exactly as decided in D45; no decision changed from test.
- **Single changes on test vs baseline (val in brackets):** Muon -0.0263 (-0.0262); 16k -0.0160 (-0.0163); 32k -0.0164 (-0.0171);
  4k +0.0094 (+0.0087); 6x768 -0.0179 (-0.0176); 19x448 +0.0212 (+0.0213). **E03 AVG4 on test:** mixture 1.3469, DCLM 1.3684,
  FineWeb-Edu 1.3947, FineWeb 1.3957: the same ordering and gaps as on validation (mixture best, DCLM better than FineWeb-Edu,
  FineWeb-Edu vs FineWeb a tie). Phase 2 sizes and Phase 3 repetition: same values + ~0.005 (rep32 3.66 = memorisation, as val).
- **Text samples (illustration only, fixed prompts, D42):** all three models write fluent, on-topic English with frequent factual
  nonsense and repetition (e.g. "Subtracting (S)" repeated by the baseline). The recipes' samples are not obviously better to the
  eye; differences of ~0.04-0.05 BPB are not visible in 3 samples. Use in the report as an honest illustration of what a ~50M model
  trained for one hour can do, not as evidence.
- **Prediction check (D42):** "test ordering = val, offsets within ~0.005, Spearman > 0.9": held (offsets within 0.0016, rho 1.000).
- **Phase 6 DONE.** No GPU work is required for the report. Credits left (estimates; check the usage pages): jinkyungjeon ~$1.5,
  giraffes573 ~$0.7, joshuahanrc ~$1.4, umarmadbi ~$0.9.
