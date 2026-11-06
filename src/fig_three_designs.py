"""Figure 2 + Table: every method across the three evaluation designs.

Usage: python fig_three_designs.py RESULTS_DIR FIG_DIR
"""
import sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

R, F = Path(sys.argv[1]), Path(sys.argv[2])
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#9a9890", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "savefig.dpi": 300})
NAME = {"Anchor-weighted adaptive LASSO": "Anchored adaptive LASSO",
        "Adaptive LASSO (no anchoring)": "Adaptive LASSO, no anchoring",
        "OASIS-2 (policy ensemble)": "OASIS-2 agent (ensemble)",
        "OASIS-2 (single signature)": "OASIS-2 agent (signature)",
        "OASIS": "OASIS agent (original)", "OASIS (full)": "OASIS agent (original)",
        "Univariate top-10 + ridge Cox": "Univariate top-10 + ridge",
        "Cox elastic net": "Cox elastic net", "Random survival forest": "Random survival forest",
        "Cox-LASSO": "Cox-LASSO", "Clinical markers (chr3 loss, 8q gain, BAP1)": "Molecular markers (chr3, 8q, BAP1)",
        "Adaptive LASSO, positional prior": "Adaptive LASSO, positional prior",
        "Adaptive LASSO, no anchoring": "Adaptive LASSO, no anchoring",
        "15-gene-class proxy": "15-gene-class proxy", "15-gene-class proxy + PRAME": "15-gene-class proxy + PRAME",
        "OASIS-2 boot (ensemble)": "OASIS-2 agent (ensemble)", "OASIS-2 boot (single)": "OASIS-2 agent (signature)"}
CONTROL = {"Adaptive LASSO, positional prior"}
ANCHORED = {"Anchored adaptive LASSO", "OASIS-2 agent (ensemble)", "OASIS-2 agent (signature)", "OASIS agent (original)"}

# ---------- TCGA CV (rep means)
rows = []
for o in json.load(open(R / "cv_results.json")):
    for m, c in o["cindex"].items():
        if m in NAME:
            rows.append((NAME[m], o["rep"], c))
for f in glob.glob(str(R / "v2_folds" / "boot_*.json")):
    o = json.load(open(f))
    rows += [(NAME["OASIS-2 boot (ensemble)"], o["rep"], o["c_ens"]), (NAME["OASIS-2 boot (single)"], o["rep"], o["c_single"])]
for o in json.load(open(R / "alasso_noanchor_tcga.json")):
    rows.append((NAME["Adaptive LASSO (no anchoring)"], o["rep"], o["cindex"]))
for o in json.load(open(R / "review_tcga.json")):
    rows += [(NAME[m], o["rep"], c) for m, c in o["cindex"].items()]
tc = pd.DataFrame(rows, columns=["method", "rep", "c"]).groupby(["method", "rep"]).c.mean().groupby("method")
tcga = pd.DataFrame({"mean": tc.mean(), "lo": tc.quantile(0.025), "hi": tc.quantile(0.975)})

# ---------- pooled CV
rows = []
for f in glob.glob(str(R / "pooled_cv" / "*.json")):
    o = json.load(open(f)); rep = int(o["split"].split("_")[0][1:])
    rows += [(NAME[m], rep, c) for m, c in o["cindex"].items() if m in NAME]
na = pd.read_csv(R / "alasso_noanchor_pooled.csv")
rows += [(NAME["Adaptive LASSO (no anchoring)"], int(s.split("_")[0][1:]), c) for s, c in zip(na.split, na.cindex)]
for o in json.load(open(R / "review_pooled.json")):
    rows += [(NAME[m], int(o["split"].split("_")[0][1:]), c) for m, c in o["cindex"].items()]
pc = pd.DataFrame(rows, columns=["method", "rep", "c"]).groupby(["method", "rep"]).c.mean().groupby("method")
pooled = pd.DataFrame({"mean": pc.mean(), "lo": pc.quantile(0.025), "hi": pc.quantile(0.975)})

# ---------- LOCO (mean over seeds per held-out cohort)
rows = []
for f in glob.glob(str(R / "pooled_loco" / "*.json")):
    o = json.load(open(f)); coh = o["split"].split("_")[1]
    rows += [(NAME[m], coh, c) for m, c in o["cindex"].items() if m in NAME]
for o in json.load(open(R / "review_loco.json")):
    rows += [(NAME[m], o["split"].split("_")[1], c) for m, c in o["cindex"].items()]
lo = pd.DataFrame(rows, columns=["method", "cohort", "c"]).groupby(["method", "cohort"]).c.mean().unstack()
loco = pd.DataFrame({"mean": lo.mean(axis=1), "lo": lo.min(axis=1), "hi": lo.max(axis=1)})

table = pd.concat({"TCGA CV": tcga["mean"], "Pooled CV": pooled["mean"]}, axis=1).join(lo).join(loco["mean"].rename("LOCO mean"))
order = table.sort_values("Pooled CV", na_position="first").index
table.loc[order[::-1]].to_csv(R / "three_designs_table.csv")
print(table.loc[order[::-1]].round(3).to_string())

# main-text figure: reinforcement-learning selectors other than the policy ensemble are in the supplement
FIG_EXCLUDE = {"OASIS agent (original)", "OASIS-2 agent (signature)"}
LABEL = {"OASIS-2 agent (ensemble)": "RL selector (policy ensemble)"}
order = [m for m in order if m not in FIG_EXCLUDE]
fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.6), sharey=True)
titles = ["TCGA 10×5-fold CV\n(79 patients)", "Pooled 10×5-fold CV\n(199 patients, 3 cohorts)",
          "Leave-one-cohort-out\n(3 held-out cohorts)"]
for ax, df, title in zip(axes, [tcga, pooled, loco], titles):
    for i, m in enumerate(order):
        if m not in df.index:
            continue
        r = df.loc[m]
        col = BLUE if m == "Anchored adaptive LASSO" else (ORANGE if m in ANCHORED else ("#1baf7a" if m in CONTROL else GRAY))
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.8, solid_capstyle="round")
        ax.scatter(r["mean"], i, s=30, color=col, zorder=3, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0.66, 0.86)
    ax.set_xticks([0.7, 0.75, 0.8, 0.85])
axes[0].set_yticks(range(len(order)), [LABEL.get(m, m) for m in order])
axes[1].set_xlabel("C-index")
fig.tight_layout()
fig.savefig(F / "fig2_three_designs.png", facecolor="white")
