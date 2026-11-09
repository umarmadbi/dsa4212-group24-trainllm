# results/runs/_replaced/

When a result file is written a second time (a deliberate rerun with `launch --fresh`, or a second `recover`), the older
file is moved here first, with a UTC timestamp added to its name. Nothing is ever deleted (DECLARATIONS section 6,
run replacement).

| File | Why it is here | Counted result |
|---|---|---|
| `E01-N11_lr2e-3__s0__20261101T160040Z.json` | Replaced: the run reused a warm container and skipped the `torch.compile` warm-up, so it got more training time than declared (infrastructure fault, D08). Rerun with `--fresh`, same seed | `../E01-N11_lr2e-3__s0.json` |
| `E01-N11_lr8e-3__s0__20261101T160537Z.json` | Replaced for the same reason (D08) | `../E01-N11_lr8e-3__s0.json` |
| `E01-N49_lr4e-3__s0__20261103T123819Z.json` | Not a rerun: the same result was recovered twice; identical to the counted file | `../E01-N49_lr4e-3__s0.json` |
| `E02-N65__s2__20261103T123735Z.json` | Not a rerun: recovered twice; identical to the counted file | `../E02-N65__s2.json` |
| `E05-N49_muon_lr1e-3_15min__s0__20261105T042208Z.json` | Not a rerun: recovered twice; identical to the counted file (the duplicate ledger row is annotated) | `../E05-N49_muon_lr1e-3_15min__s0.json` |
