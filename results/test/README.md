# results/test/

**The one-time test-set scores** (Phase 6, D47), written by `modal run modal_app.py::test_eval` (code:
`code/evaluate_test.py`). One file per model listed in [`configs/frozen_configs.yaml`](../../configs/frozen_configs.yaml):
57 files, named like the run files (`E08-N49_V16384_muon_lr2e-3_60min__s0.json`).

The test split was not used for any choice. Scoring was possible only after the frozen list was committed with status
`frozen`, from committed code, and each model could be scored only once; each file records the commit and hash of the
frozen list that allowed it. **Never recompute or edit these files.**

Each file holds, per test set: the declared BPB; the BPB without contaminated test documents; for core models, the
overlapping-window BPB (stride 128); and for marked runs, three short text samples from fixed prompts (illustration only).
Which sets were scored depends on the run's tier in the frozen list (core / data / extra).

Summary with validation side by side: `python code/summarize.py TEST`. Every validation conclusion held on test (test =
validation + about 0.005 BPB for every model; rank correlation 1.000).
