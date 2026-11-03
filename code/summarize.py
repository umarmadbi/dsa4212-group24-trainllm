"""
Summarise saved run results (results/runs/*.json) for one experiment.

    python code/summarize.py E01          # table of all E01 runs, best LR per size, checks
    python code/summarize.py E00

Needs only the Python standard library (runs on any computer, no torch).

E01 output, per model size (D07):
* every learning rate's final validation BPB (diverged runs ranked worst);
* the declared choice: lowest final BPB, flagged if at the edge of the grid;
* gap to the runner-up (compare with the 15-min seed noise when it is measured);
* a parabola fitted to BPB vs log2(LR) through the best point and its two neighbours:
  its vertex is a diagnostic of where the true optimum lies between grid points
  (NOT the selection rule, which stays "lowest BPB on the grid");
* throughput checks: a run more than 10 % slower than its size's median (the LR does not
  change the speed, so this flags a slow machine), and speed early vs late in each run
  (thermal/power throttling).
"""

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(exp: str):
    return [json.loads(f.read_text()) for f in sorted(Path("results/runs").glob(f"{exp}-*.json"))]


def bpb_key(r):
    v = r.get("final_val_bpb")
    return math.inf if (v is None or r.get("diverged") or not math.isfinite(v)) else v


def fmt_bpb(r):
    return "DIVERGED" if bpb_key(r) == math.inf else f"{r['final_val_bpb']:.4f}"


def size_of(r):
    m = re.search(r"-(N\d+)_", r["run_id"])
    c = r["config"]
    return m.group(1) if m else f"{c['n_layer']}x{c['n_embd']}"


def parabola_vertex(points):
    """points: three (log2 lr, bpb). Returns vertex log2 lr, or None if not convex."""
    (x1, y1), (x2, y2), (x3, y3) = points
    d = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d
    b = (x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1) + x1 ** 2 * (y2 - y3)) / d
    if a <= 0:
        return None
    return -b / (2 * a)


def main(exp: str):
    runs = load(exp)
    if not runs:
        print(f"no results/runs/{exp}-*.json files")
        return
    print(f"{'run':30s} {'params':>7s} {'lr':>7s} {'steps':>6s} {'tokens':>7s} {'tok/prm':>7s} "
          f"{'tok/s':>7s} {'early/late':>10s} {'clip%':>5s} {'val BPB':>9s} {'best crv':>8s} "
          f"{'gpu':>4s}")
    groups = defaultdict(list)
    for r in runs:
        c = r["config"]
        groups[size_of(r)].append(r)
        e, l = r.get("steady_tokens_per_s_early"), r.get("steady_tokens_per_s_late")
        el = f"{l / e:6.3f}" if e and l else "     -"
        clip = r.get("frac_steps_clipped")
        crv = r.get("best_curve_bpb_diagnostic")
        print(f"{r['run_id']:30s} {r['params']['total']/1e6:6.1f}M {c['lr']:7.1e} {r['steps']:6d} "
              f"{r['tokens_processed']/1e6:6.0f}M {r['tokens_processed']/r['params']['total']:7.1f} "
              f"{r['avg_tokens_per_s']:7.0f} {el:>10s} "
              f"{(100 * clip if clip is not None else float('nan')):5.1f} {fmt_bpb(r):>9s} "
              f"{(crv if crv is not None else float('nan')):8.4f} "
              f"{'L4' if 'L4' in r.get('gpu_name', '') else r.get('gpu_name', '?')[:4]:>4s}")
    if exp != "E01":
        return

    print("\nThroughput checks (LR does not change speed, so runs of one size should match):")
    flagged = False
    for size, rs in groups.items():
        med = statistics.median(r["avg_tokens_per_s"] for r in rs)
        for r in rs:
            if r["avg_tokens_per_s"] < 0.9 * med:
                flagged = True
                print(f"  SLOW: {r['run_id']} {r['avg_tokens_per_s']:.0f} tok/s vs size median "
                      f"{med:.0f} (> 10 % below: replacement rule, DECLARATIONS section 6)")
            e, l = r.get("steady_tokens_per_s_early"), r.get("steady_tokens_per_s_late")
            if e and l and l < 0.95 * e:
                flagged = True
                print(f"  SLOWED DOWN within run: {r['run_id']} late/early speed {l / e:.3f}")
    for r in runs:
        # Warm compile = the run did not pay the compile warm-up, so it trained longer
        # inside the same budget (container reuse, D08). Infrastructure fault -> rerun.
        warm = (r["config"].get("compile") and r.get("first_step_s") is not None
                and r["first_step_s"] < 5.0)
        reused = (r.get("meta") or {}).get("inputs_in_this_container", 1) > 1
        if warm or reused:
            flagged = True
            print(f"  WARM COMPILE / REUSED CONTAINER: {r['run_id']} first step "
                  f"{r.get('first_step_s')} s -> unequal budget, rerun with --fresh (D08)")
    if not flagged:
        print("  none flagged")

    print("\nBest learning rate per size (declared rule: lowest final val BPB, full FineWeb-Edu "
          "val split):")
    for size, rs in sorted(groups.items(), key=lambda kv: kv[1][0]["params"]["total"]):
        pilot_budget = min(r["budget_s"] for r in rs)          # 15-min pilots
        seed0 = [r for r in rs if r["seed"] == 0 and r["budget_s"] == pilot_budget]
        seed0.sort(key=lambda r: r["config"]["lr"])
        lrs = [r["config"]["lr"] for r in seed0]
        ranked = sorted(seed0, key=bpb_key)
        best = ranked[0]
        i = lrs.index(best["config"]["lr"])
        edge = i in (0, len(lrs) - 1)
        gap = (bpb_key(ranked[1]) - bpb_key(best)) if len(ranked) > 1 else float("nan")
        line = (f"  {size:4s} best lr {best['config']['lr']:.1e}  BPB {fmt_bpb(best)}  "
                f"(runner-up lr {ranked[1]['config']['lr']:.1e}, +{gap:.4f})")
        if not edge and all(bpb_key(seed0[j]) < math.inf for j in (i - 1, i, i + 1)):
            v = parabola_vertex([(math.log2(seed0[j]["config"]["lr"]), seed0[j]["final_val_bpb"])
                                 for j in (i - 1, i, i + 1)])
            line += (f"  parabola vertex lr {2 ** v:.1e}" if v is not None
                     else "  (no convex fit)")
        elif not edge:
            line += "  (no parabola: a neighbour diverged; optimum likely just below the "\
                    "diverging LR)" if bpb_key(seed0[i + 1]) == math.inf else ""
        if edge:
            direction = "lower" if i == 0 else "higher"
            line += f"   <-- AT EDGE OF GRID: run the next {direction} rate (x2 step)"
        print(line)
        # D09 close-call rule: when the top-2 seed-0 LRs both have >= 3 seeds, the pick is
        # the lower MEAN BPB over seeds (diverged = inf).
        allp = [r for r in rs if r["budget_s"] == pilot_budget]
        by_lr = defaultdict(list)
        for r in allp:
            by_lr[r["config"]["lr"]].append(bpb_key(r))
        multi = {lr: v for lr, v in by_lr.items() if len(v) >= 2}
        if multi:
            print("       mean over seeds: " + ", ".join(
                f"lr {lr:.1e} {statistics.mean(v):.4f} (n={len(v)}, sd {statistics.stdev(v):.4f})"
                for lr, v in sorted(multi.items()) if all(x < math.inf for x in v)))
            top2 = [ranked[0]["config"]["lr"], ranked[1]["config"]["lr"]]
            if all(len(by_lr[lr]) >= 3 for lr in top2):
                pick = min(top2, key=lambda lr: statistics.mean(by_lr[lr]))
                print(f"       CLOSE-CALL PICK (D09, mean of >=3 seeds): lr {pick:.1e}")
        others = [r for r in rs if r["seed"] != 0 and r["budget_s"] == pilot_budget]
        if others:
            same = [best] + [r for r in others if r["config"]["lr"] == best["config"]["lr"]]
            vals = [bpb_key(r) for r in same]
            sd = (f"; seed sd {statistics.stdev(vals):.4f} over {len(vals)} seeds"
                  if len(vals) >= 2 and all(v < math.inf for v in vals) else "")
            print("       extra seeds: " + ", ".join(
                f"lr {r['config']['lr']:.1e} s{r['seed']} {fmt_bpb(r)}" for r in others) + sd)
        longer = sorted((r for r in rs if r["budget_s"] != pilot_budget),
                        key=lambda r: r["config"]["lr"])
        if longer:
            print(f"       {longer[0]['budget_s'] / 60:.0f}-min horizon check: " + ", ".join(
                f"lr {r['config']['lr']:.1e} s{r['seed']} {fmt_bpb(r)}" for r in longer))


def pooled_sigma(exp="E01"):
    """Pooled within-(size, LR) seed sd over all groups with >= 2 finished seeds (D09)."""
    groups = defaultdict(list)
    for r in load(exp):
        if r["budget_s"] == 900 and bpb_key(r) < math.inf:
            groups[(size_of(r), r["config"]["lr"])].append(r["final_val_bpb"])
    ss = df = 0
    for v in groups.values():
        if len(v) >= 2:
            m = statistics.mean(v)
            ss += sum((x - m) ** 2 for x in v); df += len(v) - 1
    return (math.sqrt(ss / df), df) if df else (None, 0)


# ---------------------------------------------------------------------------
# E02: model size vs training time (D17)
REUSED_E02 = {"E01-N26_lr4e-3_60min__s0": "N26"}   # D15: counts as E02 N26 seed 0


def _quad_fit(xs, ys):
    """Least-squares y = a x^2 + b x + c (stdlib only). Returns (a, b, c)."""
    import itertools
    S = [sum(x ** k for x in xs) for k in range(5)]
    T = [sum(y * x ** k for x, y in zip(xs, ys)) for k in range(3)]
    M = [[S[4], S[3], S[2]], [S[3], S[2], S[1]], [S[2], S[1], S[0]]]
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    D = det(M)
    out = []
    for i in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][i] = T[r]
        out.append(det(Mi) / D)
    return tuple(out)


def main_e02():
    runs = load("E02")
    for rid, size in REUSED_E02.items():
        f = Path("results/runs") / f"{rid}.json"
        if f.exists():
            r = json.loads(f.read_text()); r["_size"] = size; r["_reused"] = True
            runs.append(r)
    if not runs:
        print("no E02 results yet"); return
    by = defaultdict(list)
    for r in runs:
        by[r.get("_size") or r["run_id"].split("-", 1)[1].split("__")[0]].append(r)
    # sigma60: seed sd of the N26 noise-calibration runs
    n26 = [bpb_key(r) for r in by.get("N26", []) if bpb_key(r) < math.inf]
    sigma60 = statistics.stdev(n26) if len(n26) >= 2 else None
    print(f"{'size':5s} {'params':>8s} {'lr':>7s} {'n':>2s} {'tokens':>8s} {'tok/param':>9s} "
          f"{'PFLOPs':>7s} {'BPB s0':>8s} {'mean BPB':>9s} {'sd':>7s}  notes")
    rows = []
    for size, rs in sorted(by.items(), key=lambda kv: kv[1][0]["params"]["total"]):
        r0 = next((r for r in rs if r["seed"] == 0), rs[0])
        vals = [bpb_key(r) for r in rs]
        mean = statistics.mean(vals); sd = statistics.stdev(vals) if len(vals) >= 2 else float("nan")
        P = r0["params"]["total"]; tok = statistics.mean(r["tokens_processed"] for r in rs)
        notes = []
        if any(r.get("_reused") for r in rs): notes.append("s0 = E01 horizon run (D15)")
        if any(len(r["attempts"]) > 1 for r in rs): notes.append("PREEMPTED+resumed")
        if any(r.get("meta", {}).get("git_dirty") for r in rs): notes.append("DIRTY CODE")
        ws = sorted({r.get("meta", {}).get("modal_workspace", "?") for r in rs})
        notes.append("ws " + "/".join(w.split("/")[0] for w in ws))
        rows.append((size, P, bpb_key(r0), mean, len(vals)))
        print(f"{size:5s} {P/1e6:7.1f}M {r0['config']['lr']:7.1e} {len(vals):2d} {tok/1e6:7.0f}M "
              f"{tok/P:9.1f} {r0['est_train_flops']/1e15:7.0f} {fmt_bpb(r0):>8s} {mean:9.4f} "
              f"{sd:7.4f}  {'; '.join(notes)}")
    if sigma60:
        print(f"\nsigma60 (N26, {len(n26)} seeds) = {sigma60:.4f}; single-seed threshold 2.8 sigma60 = "
              f"{2.8 * sigma60:.4f}; 3-seed threshold 1.6 sigma60 = {1.6 * sigma60:.4f}")
    ranked = sorted(rows, key=lambda t: t[3])
    best, second = ranked[0], ranked[1] if len(ranked) > 1 else None
    print(f"Best size (lowest mean BPB): {best[0]} {best[3]:.4f}"
          + (f"; runner-up {second[0]} +{second[3] - best[3]:.4f}" if second else ""))
    sizes_sorted = [t[0] for t in sorted(rows, key=lambda t: t[1])]
    if best[0] in (sizes_sorted[0], sizes_sorted[-1]):
        print("  <-- BEST SIZE IS AT THE EDGE OF THE GRID: edge rule P2 (run N4 or N92)")
    if len(rows) >= 4:
        xs = [math.log10(t[1]) for t in rows]; ys = [t[3] for t in rows]
        a, b, c = _quad_fit(xs, ys)
        if a > 0:
            xv = -b / (2 * a)
            print(f"Parabola in log10(params) through all sizes: optimum ~{10 ** xv / 1e6:.1f}M params, "
                  f"fitted BPB {a * xv * xv + b * xv + c:.4f} (diagnostic, Chinchilla 'Approach 2' style)")


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "E01"
    if exp == "E02":
        main_e02()
        sys.exit(0)
    main(exp)
    if exp == "E01":
        sg, df = pooled_sigma(exp)
        if sg:
            print(f"\nPooled 15-min seed sd (sigma15) = {sg:.4f} on {df} degrees of freedom; "
                  f"2.8 x sigma15 = {2.8 * sg:.4f} (single-seed threshold)")
