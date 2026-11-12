# E00: Pipeline checks, tokenizer and throughput benchmarks

## Phase 0 summary

**Goal:** get everything ready and proven correct before spending money on real experiments.

**What we did and what came out of it**
1. **Rules fixed first.** `DECLARATIONS.md` committed before any experiment (15:13), as the brief requires.
2. **Hardware confirmed.** Modal gives us an NVIDIA L4 with bf16 support.
3. **Data built and frozen.** Four corpora split into train / validation / test. Test and validation are locked by SHA-256 hashes. Rebuilding the data with 24x more training text gave byte-identical test, validation and tokenizer files: the pipeline is reproducible.
   - Training text per corpus: FineWeb-Edu 2.45B, FineWeb 2.53B, DCLM 2.57B tokens. Enough that no run ever reads the same text twice.
   - Leak check: training documents identical to any evaluation document were removed (5 in FineWeb-Edu, 3 in FineWeb in the full download).
4. **Our own tokenizer** (8,192 tokens) trained on training text only. About 3.9 bytes of text per token on FineWeb-Edu.
5. **Training code written and proven** with 10 automated checks at the end of Phase 0 (correct scoring, exact stop at the time limit, learning rate ends exactly at the cap, a crashed-and-resumed run gives bit-identical weights, etc.; the suite has since grown to 22, `python code/tests/test_pipeline.py`). A deliberate kill-and-resume test on Modal then found a GPU-only resume bug that the CPU tests could not see (fixed, D08/D09).
6. **Speed measured** for 6 model sizes on the L4 (the core trade-off: the 7M model reads 1.5B tokens/hour, the 124M model 0.14B).
7. **Decisions taken from the measurements:** compile on (+19 % text per hour), one batch size for all models (32,768 tokens/step), 6 sizes from 6.9M to 64.6M around the predicted optimum (~25-30M), full data download size. The speed projections later proved accurate to within 5 %; the size prediction did not (the best sizes turned out to be 36-65M). See [What happened later](#what-happened-later-predictions-vs-outcomes).

**What went wrong (and was fixed)** 
| Problem | Cause | Fix |
|---|---|---|
| Image build failed | numpy 2.5 needs Python 3.12, image had 3.11 | Image on Python 3.12, all packages checked |
| Training crashed in tests | nanoGPT's loss needs contiguous targets | `.contiguous()`; caught by our tests before the GPU |
| All 8 benchmarks crashed | YAML reads `1.0e9` as text; Modal retried each crash twice | Strict typed config loader, local validation before launch, no retries on code errors, test of every real config file |
| Results did not reach the laptop | `torch.__version__` is a torch object, not text | Results forced to plain JSON + recovery command; no data lost |
| Two of our own tests were wrong | Test design (targets = inputs; unlearnable fake data) | Fixed; lesson: a test must be able to fail for the right reason |

**Cost of Phase 0:** about $3 of Modal credit (ledger has every job; see [Time and cost](#time-and-cost-of-phase-0)).

## Where the numbers come from (reproducibility)
| What | Source in the repo | Rebuild with |
|---|---|---|
| Benchmark table, compile and micro-batch comparisons, projection check | `results/runs/E00-bench_*.json` (fields `steady_tokens_per_s`, `first_step_s`, `mfu_steady`, `peak_memory_gb`, `params`, `final_val_bpb`) | `python code/summarize.py E00` (prints this document's tables exactly) |
| Resume test | `results/runs/E00-resume_test_N7__s0.json` (`attempts` = 2) | `configs/E00/resume_test_N7.yaml`, see `docs/REPRODUCE.md` |
| Data splits, hashes, token counts, duplicate counts | `data/manifest.json`; tokenizer `data/tokenizer.json` (SHA-256 `3017aa1d...`) | `modal run --detach modal_app.py::rebuild_data`, then `modal run modal_app.py::verify_data` |
| Every job, its time and cost (including failed ones) | `results/compute_ledger.csv` (rows `E00-*`, `bench_*`, `resume_test_*`) | - |
| Correctness checks | `code/tests/test_pipeline.py` | `python code/tests/test_pipeline.py` |

**Code version disclosure.** The 8 benchmarks ran on 1 Nov from commit `5f8e548` with uncommitted changes (ledger: `+dirty`), because the training code was first committed afterwards (`3a293c3`). The code they ran differs from `3a293c3` only in how the result was sent back (the `torch.__version__` transfer bug) and in pinning Triton's default `TRITON_DEFAULT_FP_FUSION=1`; neither affects speed. Later code adds logging, guards and the resume fix; the extra per-step synchronisation costs well under 1 % (D07), so rerunning `configs/E00/bench_*.yaml` today should give speeds within machine-to-machine variation (about +-2-5 %, D08). From E01 on, `launch` refuses uncommitted code (D07). The resume test ran from committed code (`e415f8c`).

## Time and cost of Phase 0
Ledger estimates (GPU price only); Modal also bills CPU and memory, and Phase 0-1 actual spend was ~1.2x the ledger (D18 item 13).
| Job | When (1 Nov, SGT) | Container time | Counted training time | Ledger cost |
|---|---|---|---|---|
| GPU check | 17:07 | ~1 min (after ~7 min queue, not billed) | - | $0.01 |
| Data sample, 400 MB per corpus (CPU) | 17:09-17:13 | ~4 min | - | ~$0.10 |
| Benchmarks, attempt 1 (YAML bug, 8 x 3 short attempts) | 17:43 | short, est. | - | ~$0.40 |
| Benchmarks, attempt 2 (8 runs) | 18:02-18:31 | 44 min in total (307-395 s each) | 8 x 300 s = 40 min | ~$0.60 |
| Full data download, 9,500 MB per corpus (CPU) | 18:37-19:30 | ~53 min | - | ~$0.60 |
| Resume test, attempt 1 (GPU resume bug) | 20:30 | ~42 min logged incl. queue and restart | - | ~$0.56 |
| Resume test, retest (passed) | 23:27 | ~4 min | 240 s | ~$0.03 |
| **Total** | **~14:40-23:30** | **~2 GPU-hours, ~1 CPU-hour** | | **~$2.3 ledger, ~$2.8 actual** |
None of this counts toward the declared 60-minute per-run budget: the brief excludes preliminary work and tokenizer training.


## Question
Is every part of the pipeline correct, and how many tokens per hour can each model
size process on the L4? The throughput numbers decide how much data to download
(1.5x the fastest configuration's tokens per hour) and whether the size grid
brackets the compute-optimal size. Nothing here is a reported result; all GPU time is
preliminary work and goes in the ledger.

## Planned order
1. Download a small sample of each corpus (FineWeb-Edu, FineWeb, DCLM, Wikipedia) on Modal CPU.
2. Document-level splits (seed 1234), exact-duplicate removal across splits, test-split SHA-256 in `data/manifest.json`. Rerun once to confirm hashes are identical.
3. Train the 8,192-token byte-level BPE on equal samples of the three train splits only. Measure bytes per token on each validation split.
4. Correctness checks (Colab or laptop, no credits).
5. Throughput benchmarks on Modal L4, per model size, compile on and off.
6. Decide data download size, micro-batch, compile on/off; record in `docs/decisions.md` and the DECLARATIONS.md Amendments table.

## Correctness checklist
| Check | Expected | Result | Date |
|---|---|---|---|
| Initial loss at random init | about ln(8192) = 9.01 | PASS 9.018 (test first used targets = inputs and got 8.60: tied embeddings favour repeating the input; fixed test) | 2026-11-01 |
| Overfit one tiny batch | loss near 0 | PASS 0.003 after 300 steps | 2026-11-01 |
| BPB evaluator vs hand calculation on a short string | exact match | PASS (diff 7e-9, float32 rounding; batch size does not change result) | 2026-11-01 |
| `<\|endoftext\|>` targets excluded from loss and bytes | yes | PASS (counts match exactly) | 2026-11-01 |
| Each target scored exactly once (non-overlapping windows) | yes | PASS | 2026-11-01 |
| Token IDs fit uint16 and round-trip decode equals original text | yes | PASS (checked in prepare_data on real data) | 2026-11-01 |
| Save, kill, resume: counted time, tokens, RNG, data position carried over | yes | PASS locally: crash at step 25, resume from step-20 checkpoint gives bit-identical final weights to an uninterrupted run; wrong-seed resume refused | 2026-11-01 |
| Time-based cosine reaches min LR exactly at the cap (2-min test) | yes | PASS (unit test of schedule + 8 s budget run ends at min LR) | 2026-11-01 |
| Clock pauses during validation and checkpointing | yes | PASS (8 s budget with 4 curve evaluations: counted 8.0 s) | 2026-11-01 |
| Test split hash identical on rerun of the prep script | yes | PASS: test, val (all 4 corpora) and tokenizer SHA-256 identical between the 400 MB and 9,500 MB builds | 2026-11-01 |
| Kill-and-resume test on Modal itself (not just locally) | resumes, budget preserved | **FAIL 1 (2026-11-01 20:30):** kill at step 1500 worked and Modal restarted the job automatically on a new L4, but loading the checkpoint crashed (`RNG state must be a torch.ByteTensor`: map_location='cuda' moved the CPU RNG state to the GPU). Fixed (D08). **PASS on retest (2026-11-01 23:28, e415f8c):** killed at step 1500, Modal restarted it automatically, resumed from step 1265 at 120.0 s counted, finished at 240.05 s, 2 attempts. A preemption costs ~31 s of recompile inside the budget (D09) | 2026-11-02 |

## Throughput benchmarks (Modal L4)
Planned 2026-11-01: 8 runs x 5 min counted, configs in `configs/E00/`. All use context 512, 32,768 tokens per optimizer step (64 sequences), lr 6e-4, bf16.
Sizes (head dim 64): S1 6x256 (6.9M), S2 8x384 (17.5M), S3 10x512 (35.9M), S4 12x640 (64.6M), S5 12x768 (91.6M), S6 14x832 (123.5M) total params.
Extra: S3 without compile; S1 with micro-batch 64 instead of 32x2.

### Results (attempt 2, 2026-11-01 18:06-18:31; results recovered from the Volume)
| Run | Params total (non-emb) | Layers x width | Compile | Micro-batch | Compile warm-up (s) | Steady tokens/s | Tokens per 60 min (est.) | Tokens per param in 60 min | MFU | Peak mem (GB) | Val BPB after 5 min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 7.0M (4.7M) | 6x256 | on | 32x2 | 26 | 426,141 | 1.52B | 219 | 18% | 1.9 | 1.495 |
| S2 | 17.5M (14.2M) | 8x384 | on | 32x2 | 23 | 203,992 | 0.73B | 42 | 21% | 3.0 | 1.547 |
| S3 | 35.9M (31.5M) | 10x512 | on | 32x2 | 27 | 112,199 | 0.40B | 11 | 23% | 4.5 | 1.705 |
| S4 | 64.6M (59.0M) | 12x640 | on | 32x2 | 39 | 71,242 | 0.25B | 4 | 25% | 6.6 | 1.845 |
| S5 | 91.6M (85.0M) | 12x768 | on | 32x2 | 32 | 52,059 | 0.19B | 2 | 26% | 8.0 | 1.905 |
| S6 | 123.6M (116.3M) | 14x832 | on | 32x2 | 45 | 39,309 | 0.14B | 1 | 26% | 10.0 | 2.009 |
| S3_nocompile | 35.9M (31.5M) | 10x512 | off | 32x2 | 1 | 93,446 | 0.34B | 9 | 19% | 5.4 | 1.737 |
| S1_micro64 | 7.0M (4.7M) | 6x256 | on | 64x1 | 28 | 419,647 | 1.50B | 216 | 17% | 3.6 | 1.499 |

Rebuilt by `python code/summarize.py E00`. Tokens per 60 min = steady tokens/s x (3600 s - compile warm-up). Steady speed is measured between 10 % and 90 % of the run (excludes the compile warm-up). Tokens/param uses total parameters, Chinchilla's convention (Kaplan et al. count non-embedding parameters; for small models the difference is large, e.g. 32 % of S1's parameters are embeddings; E02 reports both). MFU = model FLOPs utilisation vs the L4's nominal 121 TFLOPS dense bf16 peak. The L4 runs at its 72 W power cap with clocks well below the rated maximum (D08), so MFU understates how busy the chip is; it is used only to compare sizes. Val BPB after only 5 min is NOT a result (no LR tuning, 5-min schedule); shown only as a sanity check that every size learns.

### What we learned
1. **Speed falls almost in proportion to size.** The smallest model reads ~1.5 billion tokens in an hour; the biggest only ~0.14 billion. This is the core trade-off of the project, now measured.
2. **The sweet spot is probably between S2 (17M) and S3 (36M).** Chinchilla's rule of thumb (~20 tokens per parameter) is met by S2 (~42) and S3 (~11) on either side. Chinchilla's formula with our measured compute predicts ~25-30M. S4-S6 would read only 1-4 tokens per parameter: almost certainly too big for one hour (but worth keeping one as the upper bracket, because the rule may not hold at our tiny scale - that is part of what we test).
   *Later outcome (E02): this prediction was too small. The best sizes were 36-65M (4-11 tokens/param), and the 4-tokens/param S4 (= N65) tied for best. Keeping S4 as the upper bracket is what made this visible.*
3. **Compile is worth it.** S3 with compile: +20 % tokens/s; its ~26 s warm-up costs <1 % of the hour. Net ~+19 % more text read in 60 min, and less memory.
4. **A bigger micro-batch does not help** (S1: 64x1 was 1.5 % slower than 32x2, within noise). Keep 32x2.
5. **Memory is not a constraint**: the biggest model used 10 GB of 24 GB.
6. **GPU utilisation is low for small models** (18 % for S1 vs 26 % for S6): small models leave the GPU partly idle (per-step overheads). A possible efficiency experiment later (not pursued: optimiser, shape and vocabulary were judged more informative for the budget).
   *Correction (D19): in D05 we argued this pushes the time-optimum smaller. It is the other way round: under equal time, bigger models get more FLOPs (~77 vs ~105 PFLOP for N7 vs N65 in 60 min, E02), which tilts the comparison towards bigger models. This is one reason the size prediction was too small.*
7. **The 5-minute scores already show a mini version of the trade-off**: at a 5-min budget the smallest model wins (S1 1.495 < S2 1.547 < S3 1.705). Chinchilla's formula predicts the 5-min optimum is ~8M = S1. Consistent; the 60-min experiment should shift the winner to a bigger size.
8. **(Found on re-reading the logs, 2026-11-01, D07) The bigger models slow down during a run.** Tokens/s per 30-s interval: S4 (64.6M, in our grid as N65) 73k -> 68k within 5 min (-7 %), S6 40k -> 39k (-3 %); S1-S3, S5 flat. Likely power/thermal throttling of the 72 W L4. So the "tokens per 60 min" column overestimates S4 (and maybe others over a full hour). Now monitored in every run (GPU telemetry + early/late speed).
   *Correction (D08): the 15-min E01 runs showed flat speed within a run (late/early 0.99-1.00) and the L4 at its 72 W power cap ~97-100 % of the time. The real effect is between machines: different L4s settle at different clocks (+-2.3 % tokens for the same config), which becomes part of the seed noise. The 5-min drift seen here for S4 is real in this data but small; the projection check below shows N65's real 60-min runs read 4 % less than projected.*
9. **Jobs did not all run at once**: launched 18:02, started between 18:06 and 18:26. Modal seems to have limited L4s available (or a per-account limit). Planning risk for ~40 one-hour runs: spread across the 4 accounts.

## Tokenizer statistics
| Split | Bytes per token |
|---|---|
| FineWeb-Edu val | 3.866 (train 3.867, test 3.859) |
| FineWeb val | 3.756 (train 3.751, test 3.747) |
| DCLM val | 3.632 (train 3.619, test 3.639) |
| Wikipedia val | 3.338 (test 3.367) |

Tokenizer: 8,191 BPE tokens + EOT id 8191; 7,935 merges; trained in 22 s on 3 x 200 MB.
Lower bytes/token = less compression, i.e. each token carries less text. The tokenizer compresses FineWeb-Edu best; Wikipedia (not in tokenizer training, many names/numbers) worst. Matters for the vocabulary experiment and for interpreting per-token loss (always compare BPB).

## Phase 0 data sample (2026-11-01, run E00-prepdata-400mb)
| Corpus | Revision | Test docs / MB | Val docs / MB | Train docs / MB | Train tokens | Exact dups in eval source | Train docs dropped (= eval doc) |
|---|---|---|---|---|---|---|---|
| FineWeb-Edu | 87f09149ef | 2,038 / 10.0 | 1,981 / 10.0 | 82,564 / 400 | 103.5M | 1,489 of 726,000 | 482 own, 0 cross-corpus |
| FineWeb | 9bb295ddab | 3,284 / 10.0 | 3,057 / 10.0 | 129,148 / 400 | 106.8M | 164 of 1,048,581 | 798 own, 0 cross-corpus |
| DCLM | a3b142c183 | 1,966 / 10.0 | 1,904 / 10.0 | 75,279 / 400 | 110.6M | 16 of 61,000 | 3,870 own, 0 cross-corpus |
| Wikipedia | b04c8d1ceb | 3,350 / 10.0 | 3,173 / 10.0 | - | - | 67 of 468,865 | - |

- No eval document is shared between corpora, and no exact cross-corpus train/eval duplicate was found in this sample (the FineWeb-Edu-inside-FineWeb risk did not show up as exact matches). *Near-duplicate (13-gram) check done later (D25-D26, `code/contamination.py`, `results/contamination.json`): 3.38 % of FineWeb-Edu val documents have >= 50 % of their 13-grams in some training corpus (FineWeb 1.50 %, DCLM 1.26 %, Wikipedia 0.25 %). Every comparison was re-scored without them, and on the test set removing them changed no difference by more than 0.0008 BPB (D47).*
- "Own" drops are expected: train is read from the same file the eval docs were sampled from. FineWeb-Edu train covers only the first ~11% of file 0, so ~12% of its eval docs were met and skipped (482/4,019); consistent.
- Within-train exact duplicates kept: 8 / 3 / 17 docs; negligible.
- Average doc length: FineWeb-Edu ~4.9 KB, FineWeb ~3.1 KB, DCLM ~5.3 KB, Wikipedia ~3.0 KB.
- Whole job ~4 min after image build; data prep cost is negligible, so a larger download later is cheap.


## Gaps and confounders checked
- Throughput measured with the real data loader and context 512, not synthetic tensors.
- Compile warm-up counts toward the 60 minutes, so it is measured separately per size.
- Benchmarks are 3-5 min; steady-state speed is taken after warm-up, and thermal slowdown over a full hour is checked later against final-run logs.

## Findings
Sized to the evidence: each benchmark is one 5-min run on one machine.
1. **Speed is close to inversely proportional to size** (7M: 426k tokens/s; 124M: 39k). In 60 min the 7M model reads 1.5B tokens and the 124M model 0.14B: the trade-off the project studies, measured on our hardware.
2. **The 5-min projections are reliable for 60-min planning:** the real 60-min E02 runs of the same shapes read +0.7 %, +4.5 %, +0.5 % and -4.3 % of the projected tokens (7M, 17.5M, 36M, 65M). The largest miss is the size that slowed during its benchmark.
3. **`torch.compile` pays for itself:** +20.1 % steady speed at 36M for 26 s of warm-up, net +19.2 % tokens in 60 min. One size, one run each, but the gain is far above machine-to-machine variation (+-2-5 %).
4. **A bigger micro-batch buys no speed** (-1.5 % at 7M, within machine variation), so 32 x 2 was kept.
5. **Memory is not a constraint** (10 GB of 24 GB at 124M), so model size is limited by time, not memory.
6. **The data pipeline is reproducible:** a 24x larger rebuild gave byte-identical test, validation and tokenizer files.
7. **Infrastructure failures cost more attention than compute:** five bugs in one day, all found before any reported run. Two were caught by checks we built (the CPU tests and the deliberate kill test), one at image build at no cost, and two by failed benchmark runs (~$0.40 wasted); after each failure we added a check that catches that kind of bug before the GPU. The one that mattered most, a resume crash on the GPU, would have turned every preemption of a 60-min run into a lost run.

## Limitations of the benchmarks
- **One run per configuration, 5 minutes each.** Differences under ~5 % (the micro-batch result) are within machine-to-machine variation measured later (D08).
- **Compile was measured at one size (36M) only** and assumed for the others. Every later run used it; the warm-up was 23-45 s at every size, under 1.3 % of 60 min.
- **The batch size (32,768 tokens/step) was chosen by argument, not tested** (no hardware gain from larger micro-batches; well below the critical batch size estimated by Kaplan et al. 2020 / McCandlish et al. 2018). No later experiment varied it; it is a limitation of the whole project.
- **Untuned learning rate (6e-4)** in the benchmarks, so their BPB is a sanity check only, not a comparison of sizes.
- **MFU is relative to the nominal peak**, while the L4 runs power-capped; useful for comparing sizes, not as an absolute efficiency.

## What happened later: predictions vs outcomes
| Phase 0 prediction or decision | What later experiments showed | Where |
|---|---|---|
| Optimum ~25-30M (~20 tokens/param, Chinchilla) | Best sizes 36-65M tied (4-11 tokens/param); 26M clearly worse (+0.018 BPB). Prediction too small, partly from reasoning the MFU effect in the wrong direction | E02, D19, D24 |
| Grid 6.9M-64.6M brackets the optimum | Optimum at the top of the grid; N49 (48.8M) added inside it, and the edge rule kept N65 tested with 3 seeds | E02, D19-D22 |
| Tokens per hour per size (from 5-min runs) | Accurate to +-5 % at 60 min | `summarize.py E00` projection check |
| Throughput drift within a run (S4 -7 %) | Not a within-run effect at 15-60 min; the real effect is between machines, part of seed noise (84 % of N26's seed variance, E02 P8) | D08, E02 |
| Data size 1.5x the fastest model's hour | No main-experiment run repeated data; the repetition experiment (E04) repeats it on purpose | E04 |
| Kill-and-resume works | Failed on Modal first (GPU-only bug), fixed, passed; costs ~31 s of recompile per preemption | D08, D09 |

## Sources used in Phase 0
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla), arXiv:2203.15556: ~20 tokens/param, compute-optimal size used to place the grid.
- Kaplan et al. (2020), *Scaling Laws for Neural Language Models*, arXiv:2001.08361, and McCandlish et al. (2018), *An Empirical Model of Large-Batch Training*, arXiv:1812.06162: critical batch size, used to argue 32,768 tokens/step is sample-efficient.
- Karpathy, nanoGPT (commit 3adf61e): model and training loop we adapted.
- Penedo et al. (2024), *The FineWeb Datasets*, arXiv:2406.17557, and Li et al. (2024), *DataComp-LM*, arXiv:2406.11794: the corpora.
- Pearce and Song (2024), *Reconciling Kaplan and Chinchilla Scaling Laws*, arXiv:2406.12907: why counting embedding parameters matters for small models.
- The project brief: bits per byte as a tokenizer-independent metric; preliminary work excluded from the budget.

## Surprises, failures and what we changed
- 2026-11-01: train.py / bpb.py / model.py written; 9/9 CPU checks pass (`python code/tests/test_pipeline.py`). Bugs found by the tests: (1) training targets were a non-contiguous slice and nanoGPT's loss calls `.view` -> fixed with `.contiguous()`; (2) two test-design errors (targets = inputs; unlearnable 2nd-order synthetic data) fixed. On CPU, torch.compile warm-up took 30 s of a 40 s budget for a tiny model: compile cost is counted, so it must be measured per size on the L4.
- Added gcc/g++ to the Modal image: torch.compile needs a C compiler at runtime.
- **2026-11-01 17:43, benchmark attempt 1: all 8 runs FAILED.** Cause: YAML quirk. PyYAML (YAML 1.1) reads `1.0e9` as the text "1.0e9" (an exponent needs a sign, `1.0e+9`), so `ckpt_every_s` was a string and the first checkpoint check (`time >= "1.0e9"`) crashed right after step 1. `6.0e-4` was fine because it has a sign. Why the tests missed it: they built configs in Python and never read the real YAML files.
  Second problem exposed: Modal retried each crash twice (it retries any error), so ~24 short attempts paid for a code bug. Cost small (each attempt = data load + one compiled step), est. ~$0.40 total.
  Fixes: (1) `code/config.py`: loader converts every value to its declared type and rejects bad values/unknown keys; (2) `launch` validates all configs on the local computer before any GPU starts; (3) Python errors inside a job are returned, not raised, so Modal only retries real machine loss; traceback saved to `results/failed/`; (4) new test loads every file in `configs/` (10/10 pass); (5) end-to-end CPU run of the real `bench_S1.yaml`. Failed rows kept in the ledger, annotated.
  Lesson for the report: test the real config files, not just the code; and retries should be reserved for infrastructure failures.
- **2026-11-01 ~18:00, benchmark attempt 2: training WORKS on the L4** (loss falls smoothly, LR decays to min exactly at the 300 s cap, compile warm-up ~23-27 s per run). But each result failed to arrive on the local computer: `DeserializationError ... 'torch' module is not available`. Cause: the result included `torch.__version__`, which is a torch `TorchVersion` object, not a plain string; unpickling it locally needs torch. The runs themselves completed and saved `result.json` on the `small-llm-runs` Volume before returning, so no data was lost and nothing needs rerunning.
  Fixes: versions stored as plain strings; whole result forced through JSON before returning; test asserts the result contains only plain JSON types (10/10 pass); new `recover` command fetches result.json from the Volume and records it. The ledger rows written as "failed" by attempt 2 are kept and annotated as transfer failures, with the recovered rows added.
  Also pinned `TRITON_DEFAULT_FP_FUSION=1` in the image (Triton warned its default changes in a future version) so all runs compile identically.
  Early speeds read from the console: S1 ~117M tokens / 300 s, S2 ~56M / 300 s, S3 ~13.5M in its first 147 s (incl. compile).
- 2026-11-01: `prepare_data.py` written and tested locally on fake data in the real formats (parquet, jsonl.zst): eval splits identical across reruns and train sizes; train only extends; EOT count = doc count; token bytes = UTF-8 bytes exactly; literal "<|endoftext|>" stays text. Design decisions in `decisions.md` D02.
- Found `requirements-dev.txt` saved as UTF-16 (PowerShell `>`); converted to UTF-8. Runtime versions now pinned in `requirements.txt` and used by the Modal image.
- 2026-11-01 17:06: first `prepare_data` launch failed at image build: numpy==2.5.4 requires Python >=3.12, image used 3.11. Fixed by moving the image to Python 3.12 (checked all pinned packages have cp312 wheels). No compute used. First `gpu_check` sat ~5+ min in the L4 queue (queue time not billed, not counted in budget).
- 2026-11-01 17:07: `gpu_check` passed (NVIDIA L4, bf16 supported, driver 610.57.04, CUDA 13.3) after ~7 min queue. GPU idle at 62C: watch for thermal throttling in the full-hour throughput logs (DECLARATIONS §6 slow-run rule covers it).

## Follow-ups (status at the end of the project)
- **Hugging Face token:** not added. Both downloads completed without hitting a rate limit, and `rebuild_data` works without one. Closed, not needed.
- **Do the evaluation samples mix crawl dumps?** Not checked. Validation and test come from the first file of each `sample/10BT` folder; the FineWeb samples are random samples of the whole dataset, so the first file is expected to mix crawls, but we did not read its `dump` column. Open; a minor limitation (it would affect how representative the evaluation text is, not the fairness of comparisons, since every model is scored on the same text).
- **Data size:** decided by the declared rule (1.5x the fastest model's hour, D05); no amendment needed.
- **13-gram near-duplicate check:** done (D25-D26, see the data sample section).
- **Kill-and-resume on Modal:** done (checklist).

## Full data download (2026-11-01 18:37-19:30, run E00-prepdata-9500mb)
| Corpus | Train docs | Train tokens | Bytes/token (train) | Rule >= 2.28B |
|---|---|---|---|---|
| FineWeb-Edu | 1,989,958 | 2,452,463,607 | 3.877 | OK |
| FineWeb | 3,085,610 | 2,534,238,496 | 3.753 | OK |
| DCLM | 1,728,925 | 2,565,778,440 | 3.705 | OK |

- Validation and test token counts are **identical** to the 400 MB run for every corpus (e.g. FineWeb-Edu test 2,596,414 tokens both times): the eval splits did not change when the train download grew, as designed. SHA-256 comparison: identical (below).
- DCLM's full train compresses less well (3.705 bytes/token) than its first 400 MB (3.619). Later DCLM shards differ slightly in content. Harmless (BPB normalises by bytes) but worth knowing for the data-quality experiment.
- Took ~53 min (mostly reading and hashing files; tokenizing ~8 min per corpus). Cost est. ~$0.60. HF warned about unauthenticated requests again but no rate limit was hit.

### Manifest check after full download (2026-11-01 19:53)
- Test and val SHA-256 identical to the 400 MB run for all 4 corpora; tokenizer.json SHA-256 identical (3017aa1d...). Data pipeline fully reproducible.
- Cross-corpus exact duplicates of eval docs removed from train: FineWeb-Edu 5, FineWeb 3, DCLM 0.
- **Within-train exact duplicates (kept, per declaration): FineWeb-Edu 15,229 docs (0.77 % of 1.99M), FineWeb 2,063, DCLM 748.** FineWeb-Edu's sample contains the same page from several crawls. Kept because the declared rule dedups across splits only and 0.77 % is far too small to affect size/LR comparisons; disclosed in the report. Relevant only as a footnote to the repetition experiment (a tiny amount of natural repetition exists in every run).
