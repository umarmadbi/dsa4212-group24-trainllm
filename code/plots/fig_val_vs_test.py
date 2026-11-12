"""Figure 5: validation vs test BPB for every frozen model (57)."""
import glob, json, os
import numpy as np
from _common import *

xs, ys = [], []
for p in sorted(glob.glob(os.path.join(TEST, "*.json"))):
    if p.endswith("README.md"): continue
    rid = os.path.basename(p)[:-5]
    with open(p) as f: t = json.load(f)
    with open(os.path.join(RUNS, rid + ".json")) as f: r = json.load(f)
    xs.append(r["final_val_bpb"]); ys.append(t["sets"]["fineweb_edu"]["full"]["bpb"])
xs, ys = np.array(xs), np.array(ys)
off = float(np.mean(ys - xs))
rk = lambda a: np.argsort(np.argsort(a))
rho = float(np.corrcoef(rk(xs), rk(ys))[0, 1])  # Spearman (no ties)
fig, ax = plt.subplots(figsize=(3.6, 3.2))
m = xs < 1.4
ax.scatter(xs[m], ys[m], s=14, color=C[0], linewidths=0, alpha=0.8)
lim = np.array([1.13, 1.33]); ax.plot(lim, lim + off, color=INK2, lw=1, ls="--")
ax.set_xlim(*lim); ax.set_ylim(*(lim + 0.005))
ax.set_xlabel("Validation BPB"); ax.set_ylabel("Test BPB (FineWeb-Edu)")
ax.set_title(f"{len(xs)} models: test = val {off:+.4f}, Spearman {rho:.4f}\n({(~m).sum()} heavy-repetition models off-axis)",
             fontsize=7.5, color=INK2, loc="left")
save(fig, "val_vs_test")
print("n", len(xs), "offset", off, "rho", rho, "max abs resid", np.max(np.abs(ys - xs - off)))
