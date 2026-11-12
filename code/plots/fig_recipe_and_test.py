"""Figure (report draft 2): (a) each change alone and combined vs the additive prediction (val and test);
(b) validation vs test BPB for the 57 frozen models."""
import glob, json, os
import numpy as np
from _common import *

ARMS = [("Muon\n(M)", "E05-N49_muon_lr2e-3_60min", (0, 1, 2)),
        ("16k vocab\n(V)", "E07-V16384_lr2e-3_60min", (0, 1, 2)),
        ("6x768\n(W)", "E06-W6x768_lr2e-3_60min", (0,)),
        ("RB\nM+V", "E08-N49_V16384_muon_lr2e-3_60min", (0, 1, 2)),
        ("RA\nM+W+V", "E08-W6x768_V16384_muon_lr4e-3_60min", (0, 1, 2))]
base_v = mean_val(runs("E02-N49"))
base_t = np.mean([test_bpb("E02-N49", s) for s in (0, 1, 2)])
dv, dt, n = [], [], []
for _, rid, seeds in ARMS:
    dv.append(mean_val(runs(rid, seeds)) - base_v)
    dt.append(np.mean([test_bpb(rid, s) for s in seeds]) - base_t)
    n.append(len(seeds))
add = {3: dv[0] + dv[1], 4: dv[0] + dv[1] + dv[2]}

fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(len(ARMS)); w = 0.38
err = [2 * SIGMA60 * np.sqrt(1 / k + 1 / 3) for k in n]
a.bar(x - w / 2, dv, w, color=C[0], label="validation", yerr=err, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
a.bar(x + w / 2, dt, w, color=C[2], label="test (scored once)")
for i, v in add.items():
    a.scatter([i - w / 2], [v], marker="_", s=350, color=C[7], lw=2, zorder=5,
              label="sum of single changes" if i == 3 else None)
for i, v in enumerate(dv):
    a.text(i, 0.002, f"{v:+.3f}", ha="center", va="bottom", fontsize=7, color=INK)
a.axhline(0, color=INK2, lw=1)
a.set_xticks(x); a.set_xticklabels([r[0] for r in ARMS], fontsize=8)
a.set_ylabel("BPB change vs AdamW baseline"); a.set_ylim(-0.072, 0.009)
a.legend(loc="lower left", fontsize=7)
a.set_title("(a) Muon and 16k add up; width halves", fontsize=9, loc="left")

xs, ys = [], []
for p in sorted(glob.glob(os.path.join(TEST, "*.json"))):
    rid = os.path.basename(p)[:-5]
    with open(p) as f: t = json.load(f)
    with open(os.path.join(RUNS, rid + ".json")) as f: r = json.load(f)
    xs.append(r["final_val_bpb"]); ys.append(t["sets"]["fineweb_edu"]["full"]["bpb"])
xs, ys = np.array(xs), np.array(ys)
off = float(np.mean(ys - xs))
rk = lambda q: np.argsort(np.argsort(q))
rho = float(np.corrcoef(rk(xs), rk(ys))[0, 1])
m = xs < 1.4
b.scatter(xs[m], ys[m], s=12, color=C[0], linewidths=0, alpha=0.8)
lim = np.array([1.13, 1.33]); b.plot(lim, lim + off, color=INK2, lw=1, ls="--")
b.set_xlim(*lim); b.set_ylim(*(lim + 0.005))
b.set_xlabel("Validation BPB"); b.set_ylabel("Test BPB")
b.set_title(f"(b) {len(xs)} models: test = val {off:+.4f}", fontsize=9, loc="left")
b.text(1.135, 1.32, f"Spearman {rho:.4f}", fontsize=7, color=INK2, va="top")
fig.tight_layout()
save(fig, "recipe_and_test")
