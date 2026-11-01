"""
Summarise saved run results (results/runs/*.json) for one experiment.

    python code/summarize.py E01          # table of all E01 runs, best LR per size
    python code/summarize.py E00

Needs only the Python standard library (runs on any computer, no torch).
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def load(exp: str):
    runs = []
    for f in sorted(Path("results/runs").glob(f"{exp}-*.json")):
        r = json.loads(f.read_text())
        runs.append(r)
    return runs


def main(exp: str):
    runs = load(exp)
    if not runs:
        print(f"no results/runs/{exp}-*.json files")
        return
    print(f"{'run':32s} {'params':>8s} {'lr':>8s} {'steps':>6s} {'tokens':>8s} "
          f"{'tok/s':>8s} {'val BPB':>8s} {'best curve':>10s}")
    groups = defaultdict(list)
    for r in runs:
        c = r["config"]
        m = re.search(r"-(N\d+)_", r["run_id"])
        size = m.group(1) if m else f"{c['n_layer']}x{c['n_embd']}"
        groups[size].append(r)
        print(f"{r['run_id']:32s} {r['params']['total']/1e6:7.1f}M {c['lr']:8.1e} {r['steps']:6d} "
              f"{r['tokens_processed']/1e6:7.0f}M {r['avg_tokens_per_s']:8.0f} "
              f"{r['final_val_bpb']:8.4f} {r['best_curve_bpb_diagnostic']:10.4f}")
    if exp != "E01":
        return
    print("\nBest learning rate per size (lowest final val BPB on the full FineWeb-Edu val split):")
    for size, rs in sorted(groups.items(), key=lambda kv: kv[1][0]["params"]["total"]):
        rs.sort(key=lambda r: r["config"]["lr"])
        lrs = [r["config"]["lr"] for r in rs]
        best = min(rs, key=lambda r: r["final_val_bpb"])
        edge = best["config"]["lr"] in (lrs[0], lrs[-1])
        second = sorted(rs, key=lambda r: r["final_val_bpb"])[1] if len(rs) > 1 else None
        gap = second["final_val_bpb"] - best["final_val_bpb"] if second else float("nan")
        print(f"  {size:4s} best lr {best['config']['lr']:.1e}  BPB {best['final_val_bpb']:.4f}  "
              f"(next best lr {second['config']['lr']:.1e}, +{gap:.4f})"
              + ("   <-- AT EDGE OF GRID: run one more rate beyond it" if edge else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E01")
