"""Figure 2: repeating data. (a) final BPB vs passes; (b) validation curves against passes."""
import numpy as np
from _common import *

ref = mean_val(runs("E02-N49"))
K = [2, 4, 8, 16, 32]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
xs, ms = [], []
for k in K:
    rs = runs(f"E04-N49_rep{k}")
    v = [r["final_val_bpb"] for r in rs]
    p = np.mean([r["passes_over_data"] for r in rs])
    a.scatter([p] * len(v), v, s=12, color=C[0], alpha=0.45, linewidths=0, zorder=3)
    xs.append(p); ms.append(np.mean(v))
a.plot(xs, ms, "-o", color=C[0], ms=5, zorder=4, label="N49, same data repeated")
a.axhline(ref, color=INK2, ls="--", lw=1)
a.text(2.6, ref * 1.045, f"unique data: {ref:.4f}", fontsize=7, color=INK2)
sh = run("E04-N49_rep8_shift", 0)
a.scatter([sh["passes_over_data"]], [sh["final_val_bpb"]], marker="D", s=30, color=C[1], zorder=5,
          label="8 passes, windows re-cut each pass")
a.axhline(3.462, color=C[7], ls=":", lw=1)
a.text(24, 3.462 * 0.95, "untrained model (3.46)", fontsize=7, color=C[7], ha="right", va="top")
a.set_xscale("log", base=2); a.set_yscale("log")
a.set_xticks(K); a.set_xticklabels([str(k) for k in K]); a.minorticks_off()
a.set_yticks([1.2, 1.5, 2, 3]); a.set_yticklabels(["1.2", "1.5", "2.0", "3.0"])
a.set_xlabel("Passes over the training subset in 60 min")
a.set_ylabel("Final validation BPB (log)")
a.set_title("(a) Repetition effect appears between 2 and 4 passes", fontsize=9, loc="left")
a.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=7)

for k, col in zip([4, 8, 16, 32], [C[2], C[0], C[1], C[7]]):
    r = run(f"E04-N49_rep{k}", 0)
    U = r["unique_train_tokens"]
    pts = [(c["tokens"] / U, c["val_bpb"]) for c in r["curve"] if c["tokens"] > 0]
    x, y = zip(*pts)
    b.plot(x, y, color=col, label=f"{k} passes")
    i = int(np.argmin(y)); b.scatter([x[i]], [y[i]], color=col, s=25, zorder=5, edgecolor="white")
b.set_xscale("log", base=2); b.set_ylim(1.15, 2.2)
b.set_xticks([0.5, 1, 2, 4, 8, 16, 32]); b.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32"]); b.minorticks_off()
b.axvspan(6.5, 8.5, color=GRID, alpha=0.6, zorder=0)
b.set_xlabel("Passes made so far")
b.set_ylabel("Validation BPB during training")
b.set_title("(b) Runs past ~7-8 passes start to overfit", fontsize=9, loc="left")
b.legend(fontsize=7, loc="upper right")
fig.tight_layout()
save(fig, "repetition")
