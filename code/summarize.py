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
    T = [sum(w * y * x ** k for x, y, w in zip(xs, ys, ws)) for k in range(3)]
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
        final_best = best
        if len(comp) == 1:
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
                    rng = [final_best]
                else:
                    rng = sorted([final_best] + [s for s, g in gaps.items() if g <= thr3],
                                 key=lambda s: info[s]["P"])
                    print(f"P1 RESULT: OPTIMAL RANGE {rng} (lowest mean {final_best}; gaps "
                          + ", ".join(f"{s} +{g:.4f}" for s, g in gaps.items()) + f" vs {thr3:.4f})")
                base = NOISE_SIZE if NOISE_SIZE in rng else final_best
                print(f"P9 baseline size for Phases 3-6: {base}")
        if final_best == sizes[0] and EDGE_BELOW not in by:
            print(f"  <-- P2: best size is the smallest -> run {EDGE_BELOW}")
        if final_best == sizes[-1] and EDGE_ABOVE not in by:
            print(f"  <-- P2: best size is the largest -> run {EDGE_ABOVE}")

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
        else:
            print("\nP3: parabola not convex (no interior optimum in the fitted range)")

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
