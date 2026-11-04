"""
Summarise saved run results (results/runs/*.json) for one experiment.

    python code/summarize.py E01          # table of all E01 runs, best LR per size, checks
    python code/summarize.py E00
    python code/summarize.py E02|E03|E04

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
# E02: model size vs training time (rules D17, revised D18 before any new E02 run)
REUSED_E02 = {"E01-N26_lr4e-3_60min__s0": "N26"}   # D15: counts as E02 N26 seed 0
NOISE_SIZE = "N26"                                   # 5-seed noise calibration (sigma60)
# E01 picks (DECLARATIONS Amendments 2026-11-03), for the 15-min reference points (P7)
E01_PICKS = {"N7": 2e-3, "N11": 2e-3, "N17": 4e-3, "N26": 4e-3, "N36": 4e-3, "N65": 2e-3}
EDGE_BELOW, EDGE_ABOVE, REFINE = "N4", "N92", "N49"


def _quad_fit(xs, ys, ws=None):
    """Weighted least squares y = a x^2 + b x + c (stdlib only). Returns (a, b, c)."""
    ws = ws or [1.0] * len(xs)
    S = [sum(w * x ** k for x, w in zip(xs, ws)) for k in range(5)]
    # Right-hand side must be ordered like the unknowns (a, b, c) = (x^2, x, 1) coefficients.
    # Before D19 it was ordered (1, x, x^2), which gave wrong vertices (bug from D17).
    T = [sum(w * y * x ** k for x, y, w in zip(xs, ys, ws)) for k in (2, 1, 0)]
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


def _vertex(xs, ys, ws=None):
    """Vertex x of the fitted parabola, or None if the fit is not convex / not enough points."""
    if len(xs) < 3:
        return None
    a, b, _ = _quad_fit(xs, ys, ws)
    return -b / (2 * a) if a > 0 else None


def _interp_loglog(x, xs, ys):
    """Interpolate y at x on a log-log line through sorted points (xs, ys)."""
    pts = sorted(zip(xs, ys))
    lx = math.log(x)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (lx - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
    return None


def _pooled_sd(groups):
    ss = df = 0
    for v in groups:
        v = [x for x in v if x < math.inf]
        if len(v) >= 2:
            m = statistics.mean(v)
            ss += sum((x - m) ** 2 for x in v); df += len(v) - 1
    return (math.sqrt(ss / df), df) if df else (None, 0)


def _pilot_reference():
    """Steady tokens/s per size from the E01 15-min pilots (median over all pilots of a size;
    the LR does not change speed). Reference for the slow-machine rule (DECLARATIONS sec. 6).
    Steady speed (10-90 % of the run) is used, not the average, because the compile warm-up
    is ~3 % of a 15-min pilot but ~1 % of a 60-min run."""
    ref = defaultdict(list)
    for r in load("E01"):
        if r["budget_s"] == 900 and r.get("steady_tokens_per_s"):
            ref[size_of(r)].append(r["steady_tokens_per_s"])
    return {k: statistics.median(v) for k, v in ref.items()}


def _e01_points():
    """P7: 15-min reference point per size = mean final BPB over all seeds of the E01 pick."""
    pts = {}
    for r in load("E01"):
        s = size_of(r)
        if r["budget_s"] == 900 and E01_PICKS.get(s) == r["config"]["lr"]:
            pts.setdefault(s, dict(P=r["params"]["total"], v=[]))["v"].append(bpb_key(r))
    return {s: dict(P=d["P"], bpb=statistics.mean(d["v"]), n=len(d["v"]))
            for s, d in pts.items() if all(x < math.inf for x in d["v"])}


def _instability(r):
    """P10 flag (outcome-independent of the size ranking): many clipped steps, or a training
    loss jump between consecutive 30-s log intervals after warm-up."""
    clip = r.get("frac_steps_clipped") or 0.0
    tl = [x for x in r.get("train_log", []) if x["step"] > r["config"]["warmup_steps"]]
    jump = max((b["train_loss"] - a["train_loss"] for a, b in zip(tl, tl[1:])), default=0.0)
    return (clip > 0.05 or jump > 0.10 or r.get("diverged")), clip, jump


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
    for rs in by.values():
        rs.sort(key=lambda r: r["seed"])
    sizes = sorted(by, key=lambda s: by[s][0]["params"]["total"])
    ref = _pilot_reference()

    # ---- per-size table (P4: FLOPs; tokens/param in BOTH parameter conventions) ----
    print("Tokens/param: 'tot' = all parameters (Chinchilla convention), 'ne' = non-embedding "
          "(Kaplan convention). PFLOPs = 6N+attention estimate x tokens (model.flops_per_token).")
    print(f"{'size':5s} {'params':>7s} {'non-emb':>7s} {'lr':>7s} {'n':>2s} {'tokens':>7s} "
          f"{'tok/p tot':>9s} {'tok/p ne':>8s} {'PFLOPs':>7s} {'MFU':>5s} {'BPB s0':>8s} "
          f"{'mean0-2':>8s} {'mean all':>8s} {'sd':>7s}")
    info = {}
    for s in sizes:
        rs = by[s]
        r0 = next((r for r in rs if r["seed"] == 0), None)
        v_all = [bpb_key(r) for r in rs]
        v_012 = [bpb_key(r) for r in rs if r["seed"] in (0, 1, 2)]
        P = rs[0]["params"]["total"]; Pne = rs[0]["params"]["non_embedding"]
        tok = statistics.mean(r["tokens_processed"] for r in rs)
        fl = statistics.mean(r["est_train_flops"] for r in rs)
        mfu = [r["mfu_steady"] for r in rs if r.get("mfu_steady")]
        info[s] = dict(P=P, Pne=Pne, tok=tok, s0=bpb_key(r0) if r0 else math.inf,
                       m012=statistics.mean(v_012) if v_012 else math.inf, n012=len(v_012),
                       mall=statistics.mean(v_all), n=len(v_all), vals=v_all)
        sd = statistics.stdev(v_all) if len(v_all) >= 2 and all(x < math.inf for x in v_all) \
            else float("nan")
        print(f"{s:5s} {P/1e6:6.1f}M {Pne/1e6:6.1f}M {rs[0]['config']['lr']:7.1e} {len(rs):2d} "
              f"{tok/1e6:6.0f}M {tok/P:9.1f} {tok/Pne:8.1f} {fl/1e15:7.0f} "
              f"{(statistics.mean(mfu) if mfu else float('nan')):5.3f} "
              f"{(fmt_bpb(r0) if r0 else '-'):>8s} {info[s]['m012']:8.4f} {info[s]['mall']:8.4f} "
              f"{sd:7.4f}")

    # ---- every run, with infrastructure and instability checks ----
    print("\nRuns (checks: replacement rule = crash or steady speed > 10 % below the size's E01 "
          "pilot median; warm compile / reused container = unequal budget; P10 instability):")
    flagged = False
    for s in sizes:
        for r in by[s]:
            m = r.get("meta") or {}
            notes = []
            if r.get("_reused"): notes.append("= E01 horizon run (D15)")
            tps = r.get("steady_tokens_per_s")
            if s in ref and tps:
                rel = tps / ref[s] - 1
                notes.append(f"speed {rel:+.1%} vs pilots")
                if rel < -0.10: notes.append("SLOW MACHINE -> replace (--fresh, same seed)")
            elif s not in ref:
                notes.append("no pilot speed reference (untuned edge/refinement size)")
            if r["config"].get("compile") and (r.get("first_step_s") or 99) < 5.0:
                notes.append("WARM COMPILE -> rerun (D08)")
            if m.get("inputs_in_this_container", 1) > 1:
                notes.append("REUSED CONTAINER -> rerun (D08)")
            if len(r.get("attempts", [])) > 1:
                notes.append(f"preempted+resumed ({len(r['attempts'])} attempts; kept, disclosed)")
            if m.get("git_dirty"): notes.append("DIRTY CODE")
            if "L4" not in r.get("gpu_name", ""): notes.append(f"WRONG GPU {r.get('gpu_name')}")
            if r["counted_train_s"] < r["budget_s"] and not r.get("diverged"):
                notes.append("SHORT BUDGET")
            unstable, clip, jump = _instability(r)
            if unstable:
                notes.append(f"P10 UNSTABLE (clipped {clip:.1%}, max loss jump {jump:.3f})")
            flagged |= any(k in " ".join(notes) for k in
                           ("SLOW", "WARM", "REUSED", "DIRTY", "WRONG", "SHORT", "UNSTABLE"))
            tel = r.get("gpu_telemetry_summary") or {}
            print(f"  {r['run_id']:30s} s{r['seed']} {fmt_bpb(r):>8s} {r['tokens_processed']/1e6:5.0f}M "
                  f"SM {tel.get('sm_mhz_first20', 0):4.0f}MHz "
                  f"ws {str(m.get('modal_workspace', '?')).split('/')[0]:11s} "
                  f"{str(m.get('git_commit', '?'))[:7]} code {str(m.get('code_sha256', '?'))[:8]}  "
                  + "; ".join(notes))
    print("  -> " + ("FLAGS ABOVE: resolve before applying the decision rules" if flagged
                     else "no infrastructure or instability flags"))

    # ---- noise ----
    n26 = [x for x in info.get(NOISE_SIZE, {}).get("vals", []) if x < math.inf]
    sigma60 = statistics.stdev(n26) if len(n26) >= 2 else None
    pooled, pdf = _pooled_sd([info[s]["vals"] for s in sizes])
    if sigma60:
        print(f"\nsigma60 = sd of {NOISE_SIZE} seeds ({len(n26)} seeds, {len(n26) - 1} df) = "
              f"{sigma60:.4f}; P1 single-seed threshold 2.8 sigma60 = {2.8 * sigma60:.4f}")
        print(f"pooled sigma60 over all sizes with >= 2 seeds = {pooled:.4f} ({pdf} df); "
              f"3-seed threshold 1.6 x pooled = {1.6 * pooled:.4f} (P1 step 3)")
    else:
        print(f"\nsigma60 not available yet ({len(n26)} {NOISE_SIZE} seed(s)); P1 cannot be applied")
    # P8: how much of the seed noise is machine speed?
    n26r = [r for r in by.get(NOISE_SIZE, []) if bpb_key(r) < math.inf]
    if len(n26r) >= 3:
        xs = [math.log(r["tokens_processed"]) for r in n26r]; ys = [r["final_val_bpb"] for r in n26r]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        syy = sum((y - my) ** 2 for y in ys)
        if sxx > 0 and syy > 0:
            print(f"P8 diagnostic: {NOISE_SIZE} BPB vs ln(tokens): slope {sxy / sxx:+.3f} per unit "
                  f"ln(tokens) (= {0.04 * sxy / sxx:+.4f} BPB per +4 % tokens); r^2 = "
                  f"{sxy * sxy / (sxx * syy):.2f} of the seed variance follows machine speed")

    # ---- P1 decision ----
    if sigma60:
        thr1 = 2.8 * sigma60
        ranked = sorted((s for s in sizes if info[s]["s0"] < math.inf), key=lambda s: info[s]["s0"])
        best = ranked[0]
        close = [s for s in ranked if info[s]["s0"] - info[best]["s0"] < thr1][:3]
        print(f"\nP1 step 1 (seed 0): best {best} {info[best]['s0']:.4f}; ranking "
              + ", ".join(f"{s} +{info[s]['s0'] - info[best]['s0']:.4f}" for s in ranked[1:]))
        unstable_sizes = [s for s in sizes if any(_instability(r)[0] for r in by[s])]
        # Sizes that get seeds 1,2: the close-call set, plus any P10-unstable size (the flag
        # does not depend on the ranking), so one unlucky unstable run cannot drop a size.
        comp = sorted(set(close) | set(unstable_sizes), key=lambda s: info[s]["P"])
        final_best, outright = best, False
        if len(comp) == 1:
            outright = True
            print(f"P1 DECIDED at seed 0: {best} beats every other size by > 2.8 sigma60")
        else:
            print(f"P1 step 2: close-call set (within 2.8 sigma60 of the best, at most 3) = {close}"
                  + (f"; plus P10-unstable sizes {[u for u in unstable_sizes if u not in close]}"
                     if set(unstable_sizes) - set(close) else ""))
            missing = [s for s in comp if info[s]["n012"] < 3]
            if missing:
                print(f"  -> run seeds 1,2 for: {missing}; then rerun this summary")
            else:
                thr3 = 1.6 * pooled
                final_best = min(comp, key=lambda s: info[s]["m012"])
                gaps = {s: info[s]["m012"] - info[final_best]["m012"] for s in comp if s != final_best}
                print("P1 step 3 (3-seed means, seeds 0-2): "
                      + ", ".join(f"{s} {info[s]['m012']:.4f}" for s in comp))
                if all(g > thr3 for g in gaps.values()):
                    print(f"P1 DECIDED: {final_best} (every gap > 1.6 x pooled sigma60 = {thr3:.4f})")
                    rng = [final_best]; outright = True
                else:
                    rng = sorted([final_best] + [s for s, g in gaps.items() if g <= thr3],
                                 key=lambda s: info[s]["P"])
                    print(f"P1 RESULT: OPTIMAL RANGE {rng} (lowest mean {final_best}; gaps "
                          + ", ".join(f"{s} +{g:.4f}" for s, g in gaps.items()) + f" vs {thr3:.4f})")
                base = NOISE_SIZE if NOISE_SIZE in rng else final_best
                print(f"P9 baseline size for Phases 3-6: {base}")
        # P2 as declared in D19 (before the N36/N65 seeds): an edge size is run only if the
        # smallest/largest size wins OUTRIGHT (decided at seed 0, or by > 1.6 pooled sigma).
        edge = {sizes[0]: EDGE_BELOW, sizes[-1]: EDGE_ABOVE}.get(final_best)
        if edge and edge not in by:
            print(f"  <-- P2: best size {final_best} is at the edge of the grid: "
                  + (f"won outright -> run {edge}" if outright else
                     f"but only within an optimal range -> {edge} NOT run (D19); "
                     "report that the optimum may extend beyond the grid"))

    # ---- P3: where is the optimum (diagnostic) ----
    ok = [s for s in sizes if info[s]["mall"] < math.inf]
    if len(ok) >= 4:
        xs = [math.log10(info[s]["P"]) for s in ok]; ys = [info[s]["mall"] for s in ok]
        ws = [info[s]["n"] for s in ok]
        xv = _vertex(xs, ys, ws)
        if xv is not None:
            Popt = 10 ** xv
            tok_opt = _interp_loglog(Popt, [info[s]["P"] for s in ok], [info[s]["tok"] for s in ok])
            loo = [v for i in range(len(ok)) if (v := _vertex(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:],
                                                            ws[:i] + ws[i + 1:])) is not None]
            print(f"\nP3 parabola in log10(total params), weighted by seeds, all sizes: optimum "
                  f"~{Popt / 1e6:.1f}M params"
                  + (f", ~{tok_opt / Popt:.0f} tokens/param (Chinchilla ~20)" if tok_opt else "")
                  + (f"; leave-one-size-out range {10 ** min(loo) / 1e6:.1f}-{10 ** max(loo) / 1e6:.1f}M"
                     if loo else ""))
            if 35.9e6 < Popt < 64.6e6 and REFINE not in by:
                print(f"  <-- P2b: optimum between N36 and N65 -> tune and run {REFINE} (D18)")
            if Popt > info[ok[-1]]["P"]:
                print("  note: the fitted optimum is BEYOND the largest size (extrapolated; weak)")
        else:
            print("\nP3: parabola not convex (no interior optimum in the fitted range)")
        # Post-hoc diagnostic (D19, added after seeing E02 seed 0): a parabola over a whole decade
        # of sizes is pulled by the small sizes; Chinchilla fits near the minimum. Local fit through
        # the best size by mean and its two neighbours. Labelled post hoc; never used for a decision.
        ib = min(range(len(ok)), key=lambda i: ys[i])
        if 0 < ib < len(ok) - 1:
            lv = _vertex(xs[ib - 1:ib + 2], ys[ib - 1:ib + 2])
            if lv is not None:
                tl = _interp_loglog(10 ** lv, [info[s]["P"] for s in ok], [info[s]["tok"] for s in ok])
                print(f"Post-hoc local fit ({ok[ib - 1]}, {ok[ib]}, {ok[ib + 1]}): optimum ~{10 ** lv / 1e6:.1f}M"
                      + (f", ~{tl / 10 ** lv:.1f} tokens/param" if tl else "") + " (D19, diagnostic only)")

    # ---- P7: how does the optimum move from 15 to 60 min? (diagnostic, D18) ----
    e01 = _e01_points()
    if len(e01) >= 4 and len(ok) >= 4:
        s15 = sorted(e01, key=lambda s: e01[s]["P"])
        v15 = _vertex([math.log10(e01[s]["P"]) for s in s15], [e01[s]["bpb"] for s in s15],
                      [e01[s]["n"] for s in s15])
        v60 = _vertex([math.log10(info[s]["P"]) for s in ok], [info[s]["mall"] for s in ok],
                      [info[s]["n"] for s in ok])
        if v15 is not None and v60 is not None:
            expo = (v60 - v15) / math.log10(4.0)
            print(f"P7: fitted optimum 15 min ~{10 ** v15 / 1e6:.1f}M -> 60 min ~{10 ** v60 / 1e6:.1f}M; "
                  f"N_opt ~ (time)^{expo:.2f} (Chinchilla ~0.50, Kaplan ~0.73; time-based compute, "
                  "not FLOPs)")


# ---------------------------------------------------------------------------
# E03 data quality and E04 repetition (Phase 3). Rules: D23, revised D25 before any E03/E04 run.
VAL_SETS = ["fineweb_edu", "fineweb", "dclm", "wikipedia"]
BASE_RUNS = [f"E02-N49__s{i}" for i in (0, 1, 2)]   # FineWeb-Edu / unique-data arm (P9, D22)
SIGMA_AVG4 = 0.0055      # pooled seed sd of the 4-set average, E02 sizes with >= 2 seeds (9 df; D23)
SIGMA_FWE = 0.0057       # pooled seed sd of FineWeb-Edu val BPB, E02 (10 df; D22)
SIGMA15 = 0.0067         # pooled 15-min seed sd (E01, 24 df); E03 R6 LR-check threshold 2.8 x
E04_KS = [2, 4, 8, 16, 32]
FINAL_S = 3600.0         # final runs only; 15-min pilots of E03 R6 are kept out of the arms


def _avg4(r):
    fv = r.get("final_val") or {}
    if r.get("diverged") or any(fv.get(c, {}).get("bpb") is None for c in VAL_SETS):
        return math.inf
    return statistics.mean(fv[c]["bpb"] for c in VAL_SETS)


def _set_bpb(r, c):
    v = ((r.get("final_val") or {}).get(c) or {}).get("bpb")
    return math.inf if (v is None or r.get("diverged")) else v


def _run_checks(r, ref):
    m = r.get("meta") or {}
    n = []
    # each run against ITS OWN size's pilot speed (D27: the E04 F2 runs are N17/N26/N65, not N49)
    tps, size_ref = r.get("steady_tokens_per_s"), ref.get(size_of(r))
    if tps and size_ref:
        rel = tps / size_ref - 1
        n.append(f"speed {rel:+.1%}")
        if rel < -0.10: n.append("SLOW MACHINE -> replace")
    if r["config"].get("compile") and (r.get("first_step_s") or 99) < 5.0: n.append("WARM COMPILE")
    if m.get("inputs_in_this_container", 1) > 1: n.append("REUSED CONTAINER")
    if len(r.get("attempts", [])) > 1: n.append(f"{len(r['attempts'])} attempts")
    if m.get("git_dirty"): n.append("DIRTY")
    if r["counted_train_s"] < r["budget_s"] and not r.get("diverged"): n.append("SHORT BUDGET")
    if _instability(r)[0]: n.append("UNSTABLE")
    return n


def _blocking(checks):
    """Checks that make a run unusable until replaced (declared replacement rule, sec. 6)."""
    return [c for c in checks if c.startswith(("SLOW", "WARM", "REUSED", "DIRTY", "SHORT"))]


def _load_ids(ids):
    out = []
    for rid in ids:
        f = Path("results/runs") / f"{rid}.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def _speed_slope(groups):
    """Pooled WITHIN-group slope of y on x = ln(tokens processed), and the residual sd after it.
    groups: list of [(x, y), ...], one list per configuration (>= 2 runs each). Within one
    configuration the only reason tokens differ is machine speed (same model, same 60 min), so
    this is a pure machine-speed covariate (classic ANCOVA with a common slope)."""
    sxy = sxx = 0.0
    used = []
    for g in groups:
        g = [(x, y) for x, y in g if y < math.inf]
        if len(g) >= 2:
            mx = statistics.mean(x for x, _ in g); my = statistics.mean(y for _, y in g)
            sxy += sum((x - mx) * (y - my) for x, y in g); sxx += sum((x - mx) ** 2 for x, _ in g)
            used.append((g, mx, my))
    if not used or sxx == 0:
        return None, None, 0
    beta = sxy / sxx
    ss = sum((y - my - beta * (x - mx)) ** 2 for g, mx, my in used for x, y in g)
    df = sum(len(g) - 1 for g, _, _ in used) - 1
    return beta, (math.sqrt(ss / df) if df > 0 else None), df


def _e02_groups(metric):
    """E02 sizes with >= 2 fully scored seeds: {size: [(ln tokens, metric), ...]}."""
    g = defaultdict(list)
    for r in load("E02"):
        y = metric(r)
        if y < math.inf:
            g[r["run_id"].split("-", 1)[1].split("__")[0]].append((math.log(r["tokens_processed"]), y))
    return g


def main_e03():
    ref = _pilot_reference()
    arms = {"fineweb_edu": _load_ids(BASE_RUNS)}
    pilots = []
    for r in load("E03"):
        (arms.setdefault(r["config"]["corpus"], []) if r["budget_s"] == FINAL_S else pilots).append(r)
    print("E03: which training corpus is best for N49 in 60 min? Decision metric (R1): equal-weight "
          "average of final val BPB over the four validation sets (declared). Per-set values reported too.")
    print(f"{'arm (train corpus)':20s} {'n':>2s} " + " ".join(f"{c[:11]:>11s}" for c in VAL_SETS)
          + f" {'AVG4':>8s} {'sd':>7s} {'bytes/tok':>9s} {'GB text':>7s} {'Mtok':>5s}")
    means, ntok, blocked = {}, {}, []
    for arm, rs in arms.items():
        rs.sort(key=lambda r: r["seed"])
        per = {c: statistics.mean(_set_bpb(r, c) for r in rs) for c in VAL_SETS}
        a4 = [_avg4(r) for r in rs]
        means[arm] = statistics.mean(a4)
        ntok[arm] = statistics.mean(r["tokens_processed"] for r in rs)
        sd = statistics.stdev(a4) if len(a4) >= 2 and all(x < math.inf for x in a4) else float("nan")
        # bytes of text read in training ~ tokens x bytes/token of that corpus (its val split as proxy)
        bpt = (rs[0].get("final_val") or {}).get(arm, {}).get("bytes_per_token")
        gb = (ntok[arm] * bpt / 1e9) if bpt else float("nan")
        print(f"{arm:20s} {len(rs):2d} " + " ".join(f"{per[c]:11.4f}" for c in VAL_SETS)
              + f" {means[arm]:8.4f} {sd:7.4f} {bpt or float('nan'):9.3f} {gb:7.2f} {ntok[arm] / 1e6:5.0f}")
    print("\nRuns:")
    for arm, rs in arms.items():
        for r in rs:
            ch = _run_checks(r, ref)
            if _blocking(ch): blocked.append(r["run_id"])
            print(f"  {r['run_id']:26s} s{r['seed']} AVG4 {_avg4(r):.4f} FWE {fmt_bpb(r)} "
                  f"{r['tokens_processed'] / 1e6:5.0f}M ws {str((r.get('meta') or {}).get('modal_workspace', '?')):12s} "
                  f"{str((r.get('meta') or {}).get('git_commit', '?'))[:7]}  " + "; ".join(ch))
    if blocked:
        print(f"\n!! Infrastructure flags on {blocked}: replace these runs (same seed, --fresh) "
              "before reading R1. No decision is valid until then.")
    # ---- R1 (declared): 3-seed arm means, pooled sigma of AVG4 over E02 sizes + new arms ----
    groups = {"E02-" + k: [y for _, y in v] for k, v in _e02_groups(_avg4).items()}
    for arm, rs in arms.items():
        if arm != "fineweb_edu":
            groups["E03-" + arm] = [_avg4(r) for r in rs]
    sig, df = _pooled_sd(groups.values())
    sig = sig or SIGMA_AVG4
    thr = 1.6 * sig
    print(f"\nR1 (decision, 3 seeds per arm): pooled sigma_avg4 = {sig:.4f} ({df} df; E02 alone: "
          f"{SIGMA_AVG4}); a difference in AVG4 is real if > 1.6 sigma = {thr:.4f}")
    order = sorted(means, key=means.get)
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            g = means[b] - means[a]
            ok = all(len(arms[x]) >= 3 for x in (a, b))
            print(f"  {a} better than {b} by {g:+.4f} -> " + ("REAL" if g > thr and ok else
                                                    "inconclusive" if ok else "waiting for 3 seeds"))
    # ---- R5 (declared D25, sensitivity): same comparison adjusted for machine speed ----
    sg = {"E02-" + k: v for k, v in _e02_groups(_avg4).items()}
    for arm, rs in arms.items():
        if arm != "fineweb_edu":
            sg["E03-" + arm] = [(math.log(r["tokens_processed"]), _avg4(r)) for r in rs]
    beta, sig_adj, df_adj = _speed_slope(sg.values())
    if beta is not None and sig_adj:
        xref = statistics.mean(math.log(t) for t in ntok.values())
        adj = {a: means[a] - beta * (statistics.mean(math.log(r["tokens_processed"]) for r in arms[a]) - xref)
               for a in arms}
        thr_adj = 1.6 * sig_adj
        print(f"\nR5 (sensitivity, D25): AVG4 adjusted to equal tokens processed (machine speed). Pooled "
              f"within-config slope {beta:+.4f} per unit ln(tokens), residual sd {sig_adj:.4f} ({df_adj} df), "
              f"threshold 1.6 x = {thr_adj:.4f}")
        for a in sorted(adj, key=adj.get):
            print(f"  {a:12s} adjusted AVG4 {adj[a]:.4f} (raw {means[a]:.4f}, mean tokens {ntok[a] / 1e6:.0f}M)")
        o2 = sorted(adj, key=adj.get)
        for i, a in enumerate(o2):
            for b in o2[i + 1:]:
                g = adj[b] - adj[a]
                raw = means[b] - means[a]                       # same orientation as g
                raw_v = "REAL" if raw > thr else ("REAL (reversed)" if raw < -thr else "inconclusive")
                v = "REAL" if g > thr_adj else "inconclusive"
                flip = ("" if v == raw_v else
                        f"   <- R1 says {raw_v}: report as NOT ROBUST to machine speed")
                print(f"  {a} better than {b} by {g:+.4f} -> {v}{flip}")
    # ---- R7 (declared D25, descriptive, never decisive): per-set view ----
    print("\nR7 (descriptive): per validation set, arm minus best arm on that set "
          "(pooled per-set seed sd from E02 in brackets; 1.6 x that is the per-set noise scale)")
    for c in VAL_SETS:
        sc, _ = _pooled_sd([[y for _, y in v] for v in _e02_groups(lambda r, c=c: _set_bpb(r, c)).values()])
        vals = {a: statistics.mean(_set_bpb(r, c) for r in rs) for a, rs in arms.items()}
        best = min(vals, key=vals.get)
        print(f"  {c:12s} (sd {sc or float('nan'):.4f}) best {best:12s} " +
              "  ".join(f"{a} {vals[a] - vals[best]:+.4f}" for a in arms))
    print("  Wikipedia is the only set no arm trains on (neutral). In-domain advantage of an arm = its "
          "own-corpus BPB minus the mean of the other arms on that set:")
    for a, rs in arms.items():
        if a in VAL_SETS:
            own = statistics.mean(_set_bpb(r, a) for r in rs)
            oth = [statistics.mean(_set_bpb(r, a) for r in arms[b]) for b in arms if b != a]
            if oth:
                print(f"    {a:12s} {own - statistics.mean(oth):+.4f}")
    if "mix" in arms and len(arms["mix"]) >= 1:
        mx = arms["mix"]
        fwe_mix = statistics.mean(_set_bpb(r, "fineweb_edu") for r in mx)
        fwe_arm = statistics.mean(_set_bpb(r, "fineweb_edu") for r in arms["fineweb_edu"])
        sh = [r.get("corpus_mix", {}).get("realised_share") for r in mx]
        print(f"\nR8 mixture (D26): FineWeb-Edu val {fwe_mix:.4f} vs FineWeb-Edu arm {fwe_arm:.4f} "
              f"({fwe_mix - fwe_arm:+.4f}; 3-seed noise scale 1.6 x sigma60 = {1.6 * SIGMA_FWE:.4f}); "
              f"realised token shares {sh}")
    # ---- R6 (declared D25): LR fairness pilots (15 min, seed 0, 2e-3 vs 4e-3 per corpus) ----
    print(f"\nR6 (LR check, 15-min pilots, seed 0): is 2e-3 (tuned on FineWeb-Edu) also not clearly "
          f"worse than 4e-3 on the other corpora? Inconsistent if 4e-3 is lower on the corpus's own val "
          f"set by > 2.8 sigma15 = {2.8 * SIGMA15:.4f}")
    fwe = {lr: _load_ids([f"E01-N49_lr{lr}__s{s}" for s in (0, 1, 2)]) for lr in ("2e-3", "4e-3")}
    if all(fwe.values()):
        m = {lr: statistics.mean(bpb_key(r) for r in v) for lr, v in fwe.items()}
        s0 = {lr: bpb_key(v[0]) for lr, v in fwe.items()}
        print(f"  fineweb_edu (E01): seed 0  2e-3 {s0['2e-3']:.4f}  4e-3 {s0['4e-3']:.4f}  "
              f"(3-seed means {m['2e-3']:.4f} / {m['4e-3']:.4f})")
    for corpus in ("fineweb", "dclm"):
        v = {f"{p['config']['lr']:.0e}".replace("e-0", "e-"): p for p in pilots
             if p["config"]["corpus"] == corpus and p["seed"] == 0}
        if "2e-3" in v and "4e-3" in v:
            a, b = _set_bpb(v["2e-3"], corpus), _set_bpb(v["4e-3"], corpus)
            verdict = ("INCONSISTENT -> run this arm at 4e-3, seed 0, 60 min (sensitivity, D25)"
                       if a - b > 2.8 * SIGMA15 else "consistent (2e-3 stands)")
            print(f"  {corpus:12s} own-val  2e-3 {a:.4f}  4e-3 {b:.4f}  diff(2e-3 - 4e-3) {a - b:+.4f} -> "
                  f"{verdict}; AVG4 {_avg4(v['2e-3']):.4f} / {_avg4(v['4e-3']):.4f}; "
                  + "; ".join(_run_checks(v["2e-3"], ref)[1:] + _run_checks(v["4e-3"], ref)[1:]))
        else:
            print(f"  {corpus:12s} waiting for pilots (have: {sorted(v)})")
    # ---- R3 (declared D23, made precise D25): contamination sensitivity ----
    cf = Path("results/contamination.json")
    if not cf.exists():
        print("\nR3: results/contamination.json missing -> run `modal run --detach modal_app.py::contamination` "
              "before interpreting E03.")
    else:
        cj = json.loads(cf.read_text())
        u = {k.split("/")[0]: v for k, v in cj["union_declared"].items() if k.endswith("/val")}
        need = any(v["over_1_percent"] for v in u.values())
        print(f"\nR3: contamination (>= {int(cj['declared_threshold'] * 100)} % of a doc's 13-grams in a train split; "
              f"union over training corpora): " + ", ".join(f"{c} {v['count']}/{v['docs']} ({v['share_of_docs']:.2%})"
                                                            for c, v in u.items())
              + ("" if cj.get("complete") else "  [INCOMPLETE]"))
        if not need:
            print("  no validation set above 1 %: R3 sensitivity not required (reported as such).")
        else:
            def a4_nc(r):
                f = Path("results/rescore") / f"{r['run_id']}__nocontam.json"
                if not f.exists():
                    return None
                sets = json.loads(f.read_text())["sets"]
                if any(c not in sets for c in VAL_SETS):
                    return None
                return statistics.mean((sets[c].get("excluded") or sets[c]["full"])["bpb"] for c in VAL_SETS)
            nc = {a: [a4_nc(r) for r in rs] for a, rs in arms.items()}
            miss = [r["run_id"] for a, rs in arms.items() for r, v in zip(rs, nc[a]) if v is None]
            if miss:
                print(f"  > 1 %: re-score needed (`modal run modal_app.py::rescore --exclude contamination`, "
                      f"from each run's own workspace). Missing: {miss}")
            else:
                m2 = {a: statistics.mean(v) for a, v in nc.items()}
                print("  AVG4 without contaminated documents (same documents removed for every arm; R1 threshold):")
                o3 = sorted(m2, key=m2.get)
                for i, a in enumerate(o3):
                    for b in o3[i + 1:]:
                        g = m2[b] - m2[a]
                        print(f"    {a} better than {b} by {g:+.4f} -> {'REAL' if g > thr else 'inconclusive'}"
                              f" (with all documents: {means[b] - means[a]:+.4f})")
    # ---- Follow-up trigger (declared D25): FineWeb-Edu/DCLM mixture ----
    if all(len(arms.get(x, [])) >= 3 for x in ("fineweb_edu", "dclm")):
        g = abs(means["fineweb_edu"] - means["dclm"])
        cross = []
        for c in VAL_SETS:
            sc, _ = _pooled_sd([[y for _, y in v] for v in _e02_groups(lambda r, c=c: _set_bpb(r, c)).values()])
            d = (statistics.mean(_set_bpb(r, c) for r in arms["fineweb_edu"])
                 - statistics.mean(_set_bpb(r, c) for r in arms["dclm"]))
            if sc and abs(d) > 1.6 * sc:
                cross.append("fwe" if d < 0 else "dclm")
        fire = g <= thr or ("fwe" in cross and "dclm" in cross)
        print(f"\nMixture trigger (D25): |FWE - DCLM| AVG4 = {g:.4f} (R1 threshold {thr:.4f}); per-set "
              f"winners beyond noise: {cross or 'none'} -> " +
              ("FIRES: run the 60/40 FineWeb-Edu/DCLM mixture arm, seeds 0-2" if fire else "does not fire"))


def _train_val_gap(r):
    """Final FineWeb-Edu val loss minus mean training loss over the last 5 % of the budget
    (nats/token). With unique data both are on unseen text, so the gap is ~0; repeated data
    lowers the training loss on text the model has seen, so a growing gap = memorisation."""
    tl = [x for x in r.get("train_log", []) if x["counted_s"] >= 0.95 * r["budget_s"]]
    v = ((r.get("final_val") or {}).get("fineweb_edu") or {}).get("loss_per_token")
    if not tl or v is None:
        return float("nan")
    return v - statistics.mean(x["train_loss"] for x in tl)


def _k_of(r):
    return round(290_000_000 / r["config"]["train_subset_tokens"])


def _e04_followups(follow, byk, ref):
    """D26 follow-ups (rules fixed before their runs): F2 model size on the k = 8 subset (36.25M unique
    tokens), F3 shifted windows at k = 8. Each is single seed vs E04-N49_rep8 seed 0 (single vs single:
    2 x sigma60 x sqrt(2) = 2.8 sigma60)."""
    thr = 2 * SIGMA_FWE * math.sqrt(2)
    base = next((r for r in byk.get(8, []) if r["seed"] == 0), None)
    if base is None:
        print("\nFollow-ups: waiting for E04-N49_rep8 seed 0"); return
    e02 = defaultdict(list)
    for r in load("E02"):
        e02[r["run_id"].split("-", 1)[1].split("__")[0]].append(bpb_key(r))
    b = bpb_key(base)
    print(f"\nFollow-ups (D26), all on the k = 8 subset (36.25M unique tokens), seed 0; reference E04-N49_rep8 s0 "
          f"{b:.4f}; difference real if > 2.8 sigma60 = {thr:.4f}")
    print(f"  {'run':22s} {'size':>5s} {'passes':>6s} {'FWE BPB':>8s} {'vs N49 k8':>9s} {'verdict':>14s} "
          f"{'unique-data':>11s} {'cost of limit':>13s} {'gap-ref':>8s} {'best curve@passes':>18s}  checks")
    for r in sorted(follow, key=lambda r: r["params"]["total"]):
        size = size_of(r) if not r["config"].get("train_subset_shift") else "N49"
        d = bpb_key(r) - b
        v = "BETTER" if d < -thr else ("WORSE" if d > thr else "inconclusive")
        uq = statistics.mean(e02[size]) if e02.get(size) else float("nan")
        best = min(r["curve"], key=lambda x: x["val_bpb"]) if r.get("curve") else None
        bp = f"{best['val_bpb']:.4f}@{best['tokens'] / r['unique_train_tokens']:.1f}" if best else "-"
        print(f"  {r['run_id']:22s} {size:>5s} {r['passes_over_data']:6.2f} {fmt_bpb(r):>8s} {d:+9.4f} {v:>14s} "
              f"{uq:11.4f} {bpb_key(r) - uq:+13.4f} {_train_val_gap(r) - _train_val_gap(base):+8.4f} {bp:>18s}  "
              + "; ".join(_run_checks(r, ref)[1:]))
    print("  'cost of limit' = this run minus the same size's unique-data E02 mean (how much the 36M-token limit hurts "
          "that size); gap-ref here = train-val gap minus N49 k=8's.")


def main_e04():
    ref = _pilot_reference()
    base = _load_ids(BASE_RUNS)
    if len(base) < 3:
        print("unique-data reference E02-N49 s0-2 missing"); return
    b_fwe = statistics.mean(bpb_key(r) for r in base); b_a4 = statistics.mean(_avg4(r) for r in base)
    b_gap = statistics.mean(_train_val_gap(r) for r in base)
    b_x = statistics.mean(math.log(r["tokens_processed"]) for r in base)
    thr1 = 2 * SIGMA_FWE * math.sqrt(1 + 1 / 3)
    thr3 = 1.6 * SIGMA_FWE
    beta, sig_adj, df_adj = _speed_slope(_e02_groups(bpb_key).values())
    print("E04: how much does repeating data hurt N49 in 60 min? Reference = unique data (E02-N49 s0-2: "
          f"FineWeb-Edu val {b_fwe:.4f}, AVG4 {b_a4:.4f}, train-val gap {b_gap:+.4f} nats/token).")
    print(f"R1: a single-seed k-pass run is WORSE if its FineWeb-Edu val BPB exceeds the reference by "
          f"> 2 x sigma x sqrt(1+1/3) = {thr1:.4f}; with 3 seeds (R2) by > 1.6 x sigma = {thr3:.4f}")
    print(f"{'run':20s} {'k':>3s} {'s':>1s} {'unique':>7s} {'passes':>6s} {'FWE BPB':>8s} {'delta':>8s} "
          f"{'d(speed)':>8s} {'d AVG4':>8s} {'gap-ref':>8s} {'fin-best':>8s}  checks")
    allr = [r for r in load("E04") if r["budget_s"] == FINAL_S]
    rows = [r for r in allr if r["run_id"].startswith("E04-N49_rep") and not r["config"].get("train_subset_shift")]
    follow = [r for r in allr if r not in rows]
    byk = defaultdict(list)
    for r in sorted(rows, key=lambda r: (_k_of(r), r["seed"])):
        k = _k_of(r)
        byk[k].append(r)
        d = bpb_key(r) - b_fwe
        d_adj = (d - beta * (math.log(r["tokens_processed"]) - b_x)) if beta is not None else float("nan")
        crv = r.get("best_curve_bpb_diagnostic")
        last = r["curve"][-1]["val_bpb"] if r.get("curve") else None
        print(f"{r['run_id']:20s} {k:3d} {r['seed']:1d} {r['unique_train_tokens'] / 1e6:6.1f}M "
              f"{r['passes_over_data']:6.2f} {fmt_bpb(r):>8s} {d:+8.4f} {d_adj:+8.4f} "
              f"{_avg4(r) - b_a4:+8.4f} {_train_val_gap(r) - b_gap:+8.4f} "
              f"{((last or 0) - (crv or 0)):+8.4f}  " + "; ".join(_run_checks(r, ref)))
    print("  d(speed) = delta adjusted to the reference's tokens processed (R5, sensitivity; slope "
          f"{beta if beta is None else round(beta, 4)} from E02, residual sd {sig_adj and round(sig_adj, 4)}, {df_adj} df).")
    print("  gap-ref = (val loss - train loss, last 5 %) minus the reference's: > 0 = memorising the repeated text.")
    print("  fin-best > 0 = the 1 MB validation curve went UP before the end (overfitting sign).")
    if follow:
        _e04_followups(follow, byk, ref)
    # ---- R1 seed-0 verdicts and R2 bracket (D25: robust to non-monotonic results) ----
    s0 = {k: next((r for r in v if r["seed"] == 0), None) for k, v in byk.items()}
    miss = [k for k in E04_KS if s0.get(k) is None]
    if miss:
        print(f"\nR2 waits for seed 0 of k = {miss}"); return
    worse1 = {k: bpb_key(s0[k]) - b_fwe > thr1 for k in E04_KS}
    print("\nR1 seed 0: " + ", ".join(f"k={k} {'WORSE' if worse1[k] else 'not dist.'}" for k in E04_KS))
    if not any(worse1.values()):
        print("R2: no k is worse at seed 0 (up to 32 passes): no follow-up seeds; result '> 32 passes'.")
        return
    kstar = min(k for k in E04_KS if worse1[k])
    i = E04_KS.index(kstar)
    bracket = E04_KS[max(0, i - 1):i + 1]
    nonmono = [k for k in E04_KS if k > kstar and not worse1[k]]
    if nonmono:
        print(f"  NOTE non-monotonic: k = {nonmono} not distinguishable although k = {kstar} is worse "
              "(noise; reported as such).")
    print(f"R2: smallest WORSE k* = {kstar}; bracket {bracket} gets seeds 1, 2.")

    def verdict3(k):
        v = [bpb_key(r) for r in byk.get(k, []) if r["seed"] in (0, 1, 2)]
        return (statistics.mean(v) - b_fwe > thr3, statistics.mean(v) - b_fwe) if len(v) == 3 else (None, None)
    for k in bracket:
        w, d = verdict3(k)
        print(f"  k={k}: " + ("waiting for seeds 1, 2" if w is None else
                             f"3-seed delta {d:+.4f} -> {'WORSE' if w else 'not distinguishable'}"))
    res = {k: verdict3(k)[0] for k in bracket}
    if any(v is None for v in res.values()):
        return
    lo = bracket[0] if len(bracket) == 2 else None
    step = None                                   # at most ONE extra step (D25)
    if res[kstar] and lo is not None and res[lo]:
        j = E04_KS.index(lo)
        step = E04_KS[j - 1] if j > 0 else None
        print(f"  k={lo} is also worse at 3 seeds: boundary moves DOWN one step (once)"
              + (f": k={step} gets seeds 1, 2." if step else "."))
    elif not res[kstar]:
        j = E04_KS.index(kstar)
        step = E04_KS[j + 1] if j + 1 < len(E04_KS) else None
        print(f"  k={kstar} not worse at 3 seeds (seed-0 call was noise): boundary moves UP one step (once)"
              + (f": k={step} gets seeds 1, 2." if step else "."))
    if step is not None:
        w, d = verdict3(step)
        print(f"  k={step}: " + ("waiting for seeds 1, 2" if w is None else
                                f"3-seed delta {d:+.4f} -> {'WORSE' if w else 'not distinguishable'}"))
        if w is None:
            return
    # Final call from every k that has 3 seeds (seed-0 verdicts only where no 3 seeds exist).
    v3 = {k: verdict3(k)[0] for k in E04_KS}
    final = {k: (v3[k] if v3[k] is not None else worse1[k]) for k in E04_KS}
    kw = min((k for k in E04_KS if final[k]), default=None)
    safe = max((k for k in E04_KS if not final[k] and (kw is None or k < kw)), default=None)
    print("Final verdicts (3 seeds where run, else seed 0): "
          + ", ".join(f"k={k} {'WORSE' if final[k] else 'ok'}{'' if v3[k] is None else '(3s)'}" for k in E04_KS))
    if kw is None:
        print("RESULT: no k worse up to 32 passes.")
    else:
        print(f"RESULT: repetition becomes measurably worse than unique data between {safe or 1} and {kw} "
              f"passes ({'both' if (v3[kw] is not None and (safe is None or v3[safe] is not None)) else 'not all'} "
              "ends confirmed with 3 seeds). Effect sizes with their noise are the result, not just the label.")
    print("Diagnostic (R6, D25): python code/fit_repetition.py  (Muennighoff-style effective data, R*)")


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "E01"
    if exp in ("E02", "E03", "E04"):
        {"E02": main_e02, "E03": main_e03, "E04": main_e04}[exp]()
        sys.exit(0)
    main(exp)
    if exp == "E01":
        sg, df = pooled_sigma(exp)
        if sg:
            print(f"\nPooled 15-min seed sd (sigma15) = {sg:.4f} on {df} degrees of freedom; "
                  f"2.8 x sigma15 = {2.8 * sg:.4f} (single-seed threshold)")
