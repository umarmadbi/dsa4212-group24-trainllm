"""Figure 1: final val BPB vs model size at 15 and 60 min (E01 picks, E02 finals)."""
import numpy as np
from _common import *

SIZES = ["N7", "N11", "N17", "N26", "N36", "N49", "N65"]
PICK = {"N7": "2e-3", "N11": "2e-3", "N17": "4e-3", "N26": "4e-3", "N36": "4e-3", "N49": "2e-3", "N65": "2e-3"}

def sixty(n):
    if n == "N26":  # seed 0 is the reused E01 horizon run (D15)
        return [run("E01-N26_lr4e-3_60min", 0)] + runs("E02-N26", (1, 2))
    return runs(f"E02-{n}")

fig, ax = plt.subplots(figsize=(6.2, 3.6))
for label, getter, col in [("15 min (LR pilots)", lambda n: runs(f"E01-{n}_lr{PICK[n]}"), C[1]),
                           ("60 min (declared budget)", sixty, C[0])]:
    xs, ms = [], []
    for n in SIZES:
        rs = getter(n)
        p = rs[0]["params"]["total"] / 1e6
        v = [r["final_val_bpb"] for r in rs]
        ax.scatter([p] * len(v), v, s=12, color=col, alpha=0.45, zorder=3, linewidths=0)
        xs.append(p); ms.append(np.mean(v))
        if label.startswith("60"):
            tpp = np.mean([r["tokens_processed"] for r in rs]) / rs[0]["params"]["total"]
            ax.annotate(f"{tpp:.0f}", (p, np.mean(v)), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7, color=INK2)
    ax.plot(xs, ms, "-o", color=col, ms=5, label=label, zorder=4)
    i = int(np.argmin(ms))
ax.axvline(25.6, color=INK2, ls=":", lw=1)
ax.text(25.6, ax.get_ylim()[1], " ~20 tokens/param\n (Chinchilla rule)", va="top", fontsize=7, color=INK2)
ax.set_xscale("log")
ax.set_xticks([7, 11, 17, 26, 36, 49, 65]); ax.set_xticklabels(["7", "11", "17", "26", "36", "49", "65"])
ax.minorticks_off()
ax.set_xlabel("Model size (million parameters, log scale)")
ax.set_ylabel("Validation BPB (lower is better)")
ax.legend(loc="lower left")
ax.set_title("Grey numbers: tokens read per parameter in 60 min", fontsize=8, color=INK2, loc="left")
save(fig, "size_vs_bpb")
