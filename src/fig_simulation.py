"""Figure 5: simulation, C-index gain over the no-anchoring control per setting.
Usage: python fig_simulation.py RESULTS_DIR FIG_DIR"""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

R, F = Path(sys.argv[1]), Path(sys.argv[2])
out = json.load(open(R / "simulation.json"))
d = pd.DataFrame([dict(scenario=o["scenario"], n=o["n_causal"], rep=o["rep"], method=m, c=v["c"])
                  for o in out for m, v in o["res"].items()])
w = d.pivot_table(index=["scenario", "n", "rep"], columns="method", values="c")
SC = ["S1 known risk arms", "S2 unknown regions", "S3 non-genomic drivers", "S4 mixed"]
LAB = {"S1 known risk arms": "S1\nknown\narms", "S2 unknown regions": "S2\nother\narms",
       "S3 non-genomic drivers": "S3\nnot cis", "S4 mixed": "S4\nmixed"}
M = [("anchored", "#2a78d6", "Anchored"), ("positional", "#1baf7a", "Positional prior"),
     ("permuted anchoring", "#9a9890", "Permuted anchoring")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
for ax, n in zip(axes, (5, 20)):
    for j, (m, col, lab) in enumerate(M):
        for i, s in enumerate(SC):
            g = w.loc[(s, n)]
            diff = g[m] - g["no anchoring"]
            mu, se = diff.mean(), diff.std() / np.sqrt(len(diff))
            x = i + (j - 1) * 0.22
            ax.plot([x, x], [mu - 1.96 * se, mu + 1.96 * se], color=col, lw=1.8)
            ax.scatter(x, mu, color=col, s=26, zorder=3, edgecolor="white", linewidth=1, label=lab if i == 0 else None)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(4), [LAB[s] for s in SC], fontsize=7.5)
    ax.set_title(f"{n} causal genes", loc="left", fontsize=9)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("C-index minus no-anchoring control")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
fig.savefig(F / "fig5_simulation.png", facecolor="white", bbox_inches="tight")
