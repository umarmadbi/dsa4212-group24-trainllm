# docs/experiments/

One write-up per experiment. Each follows the same structure: a summary, the question, the prediction and
design and decision rules (all written before the runs), gaps and confounders checked, the runs, results, findings, and
surprises and failures. Configs: `configs/E0x/`; results: `results/runs/E0x-*.json`; summary table:
`python code/summarize.py E0x`.

| Write-up | Experiment | Phase | Result in one line |
|---|---|---|---|
| [E00](E00_pipeline_and_benchmarks.md) | Pipeline checks and speed benchmarks | 0 | Pipeline verified; size grid 7M-65M; compile on (+20 % speed); GPU resume bug found and fixed |
| [E01](E01_learning_rate_tuning.md) | Learning rate per model size | 1 | 2e-3 or 4e-3 for every size; the pick holds at 60 minutes |
| [E02](E02_model_size.md) | Best model size in 60 minutes | 2 | N36, N49, N65 tied (4-11 tokens per parameter); N49 becomes the baseline |
| [E03](E03_data_quality.md) | Data quality | 3a | DCLM beats FineWeb-Edu and FineWeb on the four-set average; a 60/40 mix beats all three |
| [E04](E04_repetition.md) | Repeated data | 3b | 2 passes as good as fresh data, 4 passes worse, 32 passes memorise |
| [E05](E05_muon.md) | Muon vs AdamW | 4a | Muon -0.026 BPB in the same hour |
| [E06](E06_depth_width.md) | Depth vs width at the same size | 4b | 6 x 768 -0.018 (via speed); 19 x 448 +0.021 |
| [E07](E07_vocabulary.md) | Vocabulary size | 5 | 16k and 32k -0.016 to -0.017 vs 8k; 4k probably worse |
| [E08](E08_final_recipe.md) | Combined recipe and test set | 6 | RB (Muon + 16k) -0.041, RA (+ 6 x 768) -0.050 on test |
