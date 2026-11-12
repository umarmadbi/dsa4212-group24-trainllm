"""Shared helpers for the report figures. Reads only results/; never typed numbers."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(ROOT, "results", "runs")
TEST = os.path.join(ROOT, "results", "test")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

# Reference categorical palette (dataviz skill, light mode), fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SIGMA60 = 0.0057  # declared seed sd of 60-min val BPB (E02, pooled)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def run(run_id, seed):
    with open(os.path.join(RUNS, f"{run_id}__s{seed}.json")) as f:
        return json.load(f)


def runs(run_id, seeds=(0, 1, 2)):
    out = []
    for s in seeds:
        p = os.path.join(RUNS, f"{run_id}__s{s}.json")
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def mean_val(rs):
    return float(np.mean([r["final_val_bpb"] for r in rs]))


def test_bpb(run_id, seed, key="full"):
    with open(os.path.join(TEST, f"{run_id}__s{seed}.json")) as f:
        return json.load(f)["sets"]["fineweb_edu"][key]["bpb"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    print("wrote", os.path.join("report", "figures", name + ".png"))
