"""
E04 diagnostic (declared D25 before any E04 run; never used for a decision): how much is
repeated data worth, in "effective unique tokens", compared with Muennighoff et al. (2023)?

Plain language: a model that reads 290M tokens made of a 72M-token slice repeated 4 times ends
with some loss L. We ask: how many tokens of NEW data would N49 have needed to reach the same L?
That number is the effective data D'. If repetition were as good as new data, D' = 290M; if
repeats were worthless, D' = the 72M unique tokens.

Method:
1. Loss law L(N, D) = E + A/N^a + B/D^b fitted to our unique-data runs (code/fit_scaling_law.py,
   14 points: 7 sizes x 15 and 60 min). Hold N = N49 and invert: D' = (B / (L - E - A/N^a))^(1/b).
2. Muennighoff et al. (2023, eq. 5): D' = U + U * R* * (1 - exp(-R / R*)), with U the unique
   tokens and R = passes - 1 the number of REPEATED passes. R* is the "half-life" of repeats
   (their fit: R*_D ~ 15.4). Fit R* by least squares on log D' over the E04 runs.
3. Report D'/D (fraction of seen tokens that counted as new) per k, and R* with a leave-one-out range.

    python code/fit_repetition.py           (needs numpy and scipy; reads results/runs)

Caveats (state with the numbers): the loss law is a 5-parameter fit to 14 points and is
extrapolated in D only downwards (D' <= 290M, inside the fitted range: 15-min runs reach ~70M);
L close to the law's floor makes D' unstable (flagged); 1 seed per k except the R2 bracket.
"""
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from fit_scaling_law import fit, points

MUENNIGHOFF_R_STAR = 15.4
BASE = [f"E02-N49__s{i}" for i in (0, 1, 2)]


def runs():
    out = []
    for f in sorted(Path("results/runs").glob("E04-*.json")):
        r = json.loads(f.read_text())
        if r["budget_s"] == 3600 and not r.get("diverged"):
            out.append(r)
    return out


def d_eff(L, N, E, A, a, B, b):
    rest = L - E - A / N ** a
    return (B / rest) ** (1 / b) if rest > 0 else math.inf


def muennighoff(U, R, rs):
    return U + U * rs * (1 - math.exp(-R / rs))


def fit_rstar(rows):
    def obj(lr):
        rs = math.exp(lr)
        return sum((math.log(muennighoff(U, R, rs)) - math.log(De)) ** 2 for U, R, De in rows)
    res = minimize_scalar(obj, bounds=(math.log(0.05), math.log(1e4)), method="bounded")
    return math.exp(res.x)


def main():
    rows = points()
    N_, D_, L_, w_ = (np.array([r[k] for r in rows], float) for k in ("N", "D", "L", "n"))
    E, A, a, B, b = fit(N_, D_, L_, w_)
    base = [json.loads((Path("results/runs") / f"{x}.json").read_text()) for x in BASE]
    N = base[0]["params"]["total"]
    print(f"Loss law (unique data, 14 points): L = {E:.3f} + {A:.3g}/N^{a:.3f} + {B:.3g}/D^{b:.3f}; "
          f"N49 floor E + A/N^a = {E + A / N ** a:.4f} BPB")
    for r in base:
        De = d_eff(r["final_val_bpb"], N, E, A, a, B, b)
        print(f"  calibration {r['run_id']}: seen {r['tokens_processed'] / 1e6:.0f}M, "
              f"effective {De / 1e6:.0f}M (should be ~1x)")
    data = []
    print(f"\n{'run':20s} {'unique':>7s} {'passes':>6s} {'BPB':>7s} {'D seen':>7s} {'D eff':>7s} {'eff/seen':>8s}")
    for r in sorted(runs(), key=lambda r: -r["unique_train_tokens"]):
        U, D = r["unique_train_tokens"], r["tokens_processed"]
        De = d_eff(r["final_val_bpb"], N, E, A, a, B, b)
        flag = "" if math.isfinite(De) else "  (loss above the law's reach: D' undefined, excluded)"
        print(f"{r['run_id']:20s} {U / 1e6:6.1f}M {D / U:6.2f} {r['final_val_bpb']:7.4f} {D / 1e6:6.0f}M "
              f"{De / 1e6:6.0f}M {De / D:8.2f}{flag}")
        if math.isfinite(De) and D / U > 1:
            data.append((U, D / U - 1, min(De, D)))   # D' cannot exceed the tokens seen
    if len(data) < 3:
        print("\nToo few usable runs to fit R*."); return
    rs = fit_rstar(data)
    loo = [fit_rstar(data[:i] + data[i + 1:]) for i in range(len(data))]
    print(f"\nMuennighoff fit: R* = {rs:.1f} repeated passes (leave-one-out {min(loo):.1f}-{max(loo):.1f}); "
          f"their R*_D ~ {MUENNIGHOFF_R_STAR}. Larger R* = repeats keep their value longer.")
    for U, R, De in data:
        print(f"  passes {R + 1:5.1f}: measured D'/U {De / U:6.2f}, fitted {muennighoff(U, R, rs) / U:6.2f}, "
              f"Muennighoff R*=15.4 {muennighoff(U, R, MUENNIGHOFF_R_STAR) / U:6.2f}")


if __name__ == "__main__":
    main()
