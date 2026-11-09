# results/failed/

Python error traces of training runs that crashed, written by `modal_app.py launch`. The run keeps its `failed` row in
`../compute_ledger.csv`.

| File | What happened |
|---|---|
| `E00-resume_test_N7__s0.txt` | The kill-and-resume test (E00): Modal restarted the deliberately killed job as intended, but resuming crashed on the GPU (`TypeError: RNG state must be a torch.ByteTensor`: the saved random-number states were loaded onto the GPU). The CPU tests could not catch it. Fixed in D08 and rerun with `--fresh`; the rerun passed (`../runs/E00-resume_test_N7__s0.json`) |
