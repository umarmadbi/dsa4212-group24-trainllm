"""Figure 4: each change alone and combined, val and test, vs the additive prediction."""
import numpy as np
from _common import *

ARMS = [("Muon (M)", "E05-N49_muon_lr2e-3_60min", (0, 1, 2)),
        ("16k vocab (V)", "E07-V16384_lr2e-3_60min", (0, 1, 2)),
        ("6x768 (W)", "E06-W6x768_lr2e-3_60min", (0,)),
        ("RB = M+V", "E08-N49_V16384_muon_lr2e-3_60min", (0, 1, 2)),
        ("RA = M+W+V", "E08-W6x768_V16384_muon_lr4e-3_60min", (0, 1, 2))]
base_v = mean_val(runs("E02-N49"))
base_t = np.mean([test_bpb("E02-N49", s) for s in (0, 1, 2)])
dv, dt, n = [], [], []
for _, rid, seeds in ARMS:
    dv.append(mean_val(runs(rid, seeds)) - base_v)
    dt.append(np.mean([test_bpb(rid, s) for s in seeds]) - base_t)
    n.append(len(seeds))
add = {3: dv[0] + dv[1], 4: dv[0] + dv[1] + dv[2]}
fig, ax = plt.subplots(figsize=(6.4, 3.3))
x = np.arange(len(ARMS)); w = 0.36
err = [2 * SIGMA60 * np.sqrt(1 / k + 1 / 3) for k in n]
ax.bar(x - w / 2, dv, w, color=C[0], label="validation", yerr=err, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
ax.bar(x + w / 2, dt, w, color=C[2], label="test (scored once)")
for i, v in add.items():
    ax.scatter([i - w / 2], [v], marker="_", s=400, color=C[7], lw=2, zorder=5,
               label="sum of single changes" if i == 3 else None)
for i, v in enumerate(dv):
    ax.text(i - w / 2, -0.0015, f"{v:+.3f}", ha="center", va="top", fontsize=6.5, color="white", fontweight="bold")
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(x); ax.set_xticklabels([a[0] for a in ARMS])
ax.set_ylabel("BPB change vs AdamW baseline")
ax.set_ylim(-0.075, 0.004)
ax.legend(loc="lower left", fontsize=7)
ax.set_title("Error bars: 2 SE from the declared seed noise. W has one seed.", fontsize=8, color=INK2, loc="left")
save(fig, "recipe_attribution")
