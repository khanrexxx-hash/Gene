"""Selection stability over the 50 TCGA CV folds: Nogueira et al. (2018) Phi and Jaccard.

Phi is chance-corrected and not biased by signature size. p = number of genes after filtering.
Usage: python stability.py RESULTS_DIR P_FEATURES [FIG_DIR]
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path

R, p = Path(sys.argv[1]), int(sys.argv[2])


def phi(sets, p):
    m = len(sets)
    allg = sorted({g for s in sets for g in s})
    Z = np.zeros((m, len(allg)))
    ix = {g: i for i, g in enumerate(allg)}
    for r, s in enumerate(sets):
        Z[r, [ix[g] for g in s]] = 1
    pf = Z.mean(0)
    s2 = m / (m - 1) * pf * (1 - pf)  # unbiased per-feature variance; zero for never-selected genes
    kbar = Z.sum(1).mean()
    return 1 - (s2.sum() / p) / ((kbar / p) * (1 - kbar / p))


def jac(sets):
    s = [set(x) for x in sets if len(x)]
    return float(np.mean([len(a & b) / len(a | b) for i, a in enumerate(s) for b in s[i + 1:]]))


S = {}
v1 = json.load(open(R / "cv_results.json"))
for m, nm in [("Anchor-weighted adaptive LASSO", "Anchored adaptive LASSO"), ("Cox-LASSO", "Cox-LASSO"),
              ("Cox elastic net", "Cox elastic net"), ("Univariate top-10 + ridge Cox", "Univariate top-10 + ridge Cox"),
              ("OASIS (full)", "OASIS agent (original)"), ("OASIS w/o anchoring prior", "OASIS agent, no anchoring prior"),
              ("OASIS with random prior", "OASIS agent, random prior")]:
    S[nm] = [o["sets"][m] for o in v1]
S["Adaptive LASSO, no anchoring"] = [o["genes"] for o in json.load(open(R / "alasso_noanchor_tcga.json"))]
S["Adaptive LASSO, positional prior"] = [o["sets"]["Adaptive LASSO, positional prior"] for o in json.load(open(R / "review_tcga.json"))]
S["OASIS-2 agent (single signature)"] = [json.load(open(f))["sel"] for f in sorted(glob.glob(str(R / "v2_folds" / "boot_*.json")))]
rows = [dict(method=k, folds=len(v), mean_genes=np.mean([len(x) for x in v]), phi=phi(v, p), jaccard=jac(v))
        for k, v in S.items()]
t = pd.DataFrame(rows).sort_values("phi", ascending=False)
t.to_csv(R / "stability_table.csv", index=False)
print(t.round(3).to_string(index=False))

# ---------- Figure 3: signature size vs Phi (written after analyze.py, replacing its Jaccard version)
if len(sys.argv) > 3:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = Path(sys.argv[3])
    BLUE, ORANGE, GREEN, GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#52514e"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": INK2, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for _, r in t[~t.method.str.startswith("OASIS")].iterrows():   # RL selectors: Supplementary Table S4
        m = r.method
        col = BLUE if m == "Anchored adaptive LASSO" else GREEN if "positional" in m else ORANGE if m.startswith("OASIS") else GRAY
        ax.scatter(r.mean_genes, r.phi, s=46, color=col, edgecolor="white", linewidth=1.4, zorder=3)
        right = r.mean_genes < 30
        dy = {"OASIS agent (original)": -9, "Adaptive LASSO, no anchoring": -10, "Adaptive LASSO, positional prior": -3}.get(m, 4)
        ax.annotate(m, (r.mean_genes, r.phi), xytext=(6 if right else -6, dy), textcoords="offset points",
                    fontsize=7, color=INK2, ha="left" if right else "right")
    ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"]); ax.minorticks_off()
    ax.set_xlim(4, 90); ax.set_ylim(0.05, 0.5)
    ax.set_xlabel("Mean signature size (genes, log scale)")
    ax.set_ylabel("Selection stability, Φ (Nogueira)")
    ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(F / "fig3_stability.png", facecolor="white")
