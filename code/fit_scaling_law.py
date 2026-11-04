"""
Post-hoc diagnostic (D24, added after E02; never used for a decision): fit a Chinchilla-style
loss law  L(N, D) = E + A / N^alpha + B / D^beta  to our own completed runs, and ask where the
optimum would be at EQUAL FLOPs (not equal time).

Data points: every size at two horizons, both with a complete cosine schedule:
  * 15 min: E01 pilots at the LR finally picked for that size (mean over its seeds)
  * 60 min: E02 runs (mean over seeds 0-2; N26 includes the reused E01 horizon run as seed 0)
N = total parameters, D = tokens processed, L = final FineWeb-Edu validation BPB.
Fitting as in Hoffmann et al. (2022, Approach 3): Huber loss (delta 1e-3) on log-loss
residuals, L-BFGS from a grid of starts; points weighted by their number of seeds.
Compute-optimal allocation for a FLOP budget C (C = 6 N D):
  N_opt = G (C/6)^(beta/(alpha+beta)),  G = (alpha A / (beta B))^(1/(alpha+beta)).

    python code/fit_scaling_law.py          (needs numpy and scipy; reads results/runs)

Caveats (state them with the numbers): 14 points and 5 parameters; only two training lengths
per size; 15-min runs carry a larger share of warm-up and compile time; the fitted form is
assumed, not tested.
"""
import glob
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PICKS = {"N7": 2e-3, "N11": 2e-3, "N17": 4e-3, "N26": 4e-3, "N36": 4e-3, "N49": 2e-3, "N65": 2e-3}
REUSED = "E01-N26_lr4e-3_60min__s0"      # counts as E02 N26 seed 0 (D15)
CHINCHILLA = {"Hoffmann et al. 2022 (Approach 3)": (406.4, 410.7, 0.34, 0.28),
              "Besiroglu et al. 2024 (refit)": (482.01, 2085.43, 0.3478, 0.3658)}


def points():
    pts = {}
    for f in glob.glob("results/runs/E0[12]-*.json"):
        r = json.loads(Path(f).read_text())
        rid = r["run_id"]
        if rid == REUSED:
            key = ("N26", 60)
        elif rid.startswith("E01-"):
            size = rid.split("-")[1].split("_")[0]
            if r["budget_s"] != 900 or PICKS.get(size) != r["config"]["lr"]:
                continue
            key = (size, 15)
        elif rid.startswith("E02-"):
            key = (rid.split("-", 1)[1].split("__")[0], 60)
        else:
            continue
        if r["seed"] > 2 or r.get("diverged"):
            continue
        pts.setdefault(key, []).append(r)
    rows = []
    for (size, h), rs in sorted(pts.items()):
        rows.append(dict(size=size, minutes=h, n=len(rs), N=rs[0]["params"]["total"],
                         D=st.mean(r["tokens_processed"] for r in rs),
                         L=st.mean(r["final_val_bpb"] for r in rs),
                         C=st.mean(r["est_train_flops"] for r in rs)))
    return rows


def huber(x, d=1e-3):
    a = np.abs(x)
    return np.where(a < d, 0.5 * x * x, d * (a - 0.5 * d))


def fit(N, D, L, w):
    best = None
    for a0 in (0.2, 0.35, 0.5, 0.7):
        for b0 in (0.2, 0.35, 0.5, 0.7):
            for lA in (3, 6, 9):
                for lB in (3, 6, 9):
                    def obj(p):
                        E, la, lb, a, b = p
                        pred = E + np.exp(la) / N ** a + np.exp(lb) / D ** b
                        return np.sum(w * huber(np.log(pred) - np.log(L)))
                    res = minimize(obj, [0.8, lA, lB, a0, b0], method="L-BFGS-B",
                                   bounds=[(0, 1.2), (-5, 30), (-5, 30), (0.05, 1.5), (0.05, 1.5)])
                    if best is None or res.fun < best.fun:
                        best = res
    E, la, lb, a, b = best.x
    return E, np.exp(la), a, np.exp(lb), b


def optimum(A, B, a, b, C):
    G = (a * A / (b * B)) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b))
    return N, C / 6 / N / N        # (N_opt, tokens per parameter)


def main():
    rows = points()
    N, D, L, w = (np.array([r[k] for r in rows], float) for k in ("N", "D", "L", "n"))
    E, A, a, B, b = fit(N, D, L, w)
    print(f"Points: {len(rows)} (sizes x {{15, 60}} min)")
    print(f"Fit: L = {E:.3f} + {A:.3g}/N^{a:.3f} + {B:.3g}/D^{b:.3f}   "
          f"max |residual| {np.max(np.abs(E + A / N ** a + B / D ** b - L)):.4f} BPB")
    C = st.mean(r["C"] for r in rows if r["minutes"] == 60)
    No, ratio = optimum(A, B, a, b, C)
    print(f"Equal-FLOP optimum at our mean 60-min budget C = {C:.2e} FLOPs: N ~ {No / 1e6:.0f}M, "
          f"~{ratio:.1f} tokens/param")
    loo = []
    for i in range(len(rows)):
        m = np.arange(len(rows)) != i
        _, A_, a_, B_, b_ = fit(N[m], D[m], L[m], w[m])
        loo.append(optimum(A_, B_, a_, b_, C))
    print(f"  leave-one-point-out: N {min(x[0] for x in loo) / 1e6:.0f}-{max(x[0] for x in loo) / 1e6:.0f}M, "
          f"{min(x[1] for x in loo):.1f}-{max(x[1] for x in loo):.1f} tokens/param")
    for name, (A_, B_, a_, b_) in CHINCHILLA.items():
        n_, r_ = optimum(A_, B_, a_, b_, C)
        print(f"{name} law at the same C: N ~ {n_ / 1e6:.0f}M, ~{r_:.0f} tokens/param")


if __name__ == "__main__":
    main()
