"""Figure 3: Muon vs AdamW at N49: gap along the hour, at equal time and at equal tokens."""
import numpy as np
from _common import *

def curves(rid):
    rs = runs(rid)
    t = np.array([c["counted_s"] for c in rs[0]["curve"]]) / 60
    tok = np.mean([[c["tokens"] for c in r["curve"]] for r in rs], axis=0)
    bpb = np.mean([[c["val_bpb"] for c in r["curve"]] for r in rs], axis=0)
    return t, tok, bpb

ta, ka, ya = curves("E02-N49")
tm, km, ym = curves("E05-N49_muon_lr2e-3_60min")
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
s = slice(2, None)
a.plot(ta[s], ya[s], color=C[1], label="AdamW (3 seeds)")
a.plot(tm[s], ym[s], color=C[0], label="Muon (3 seeds)")
a.set_xlabel("Counted training time (min)"); a.set_ylabel("Validation BPB (1 MB curve subset)")
a.set_title("(a) Learning curves, N49, 60 min", fontsize=9, loc="left"); a.legend()
gap_t = ym - ya
gap_k = ym - np.interp(km, ka, ya)  # AdamW read at Muon's token count
b.axhline(0, color=INK2, lw=1)
b.plot(tm[1:], gap_t[1:], "-o", ms=3, color=C[0], label="at equal time (declared)")
b.plot(tm[1:], gap_k[1:], "--", color=C[2], label="at equal tokens (per step)")
b.set_ylim(-0.21, 0.01)
b.set_xlabel("Counted training time (min)"); b.set_ylabel("Muon minus AdamW (BPB)")
b.set_title("(b) Gain is large early, then settles near 0.03", fontsize=9, loc="left"); b.legend(loc="lower right")
fig.tight_layout()
save(fig, "muon_curves")
