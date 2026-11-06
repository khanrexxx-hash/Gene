"""Statistics and figures from cv_results.json (and external results if present).

Usage: python analyze.py RESULTS_DIR FIG_DIR
"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

R, F = Path(sys.argv[1]), Path(sys.argv[2])
F.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9890"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300})

out = json.load(open(R / "cv_results.json"))
if (R / "cv_results_plus.json").exists():  # merge the OASIS+ run (same folds)
    plus = {(o["rep"], o["fold"]): o for o in json.load(open(R / "cv_results_plus.json"))}
    for o in out:
        p = plus[(o["rep"], o["fold"])]
        o["cindex"].update(p["cindex"]); o["sets"].update(p["sets"])
        o["history_plus"] = p["history"]
ref = sys.argv[3] if len(sys.argv) > 3 else "OASIS (full)"

rows = [dict(rep=o["rep"], fold=o["fold"], method=m, cindex=c) for o in out for m, c in o["cindex"].items()]
df = pd.DataFrame(rows)
df.to_csv(R / "cv_fold_cindex_all.csv", index=False)


def jaccard_stability(set_list):
    s = [set(x) for x in set_list if len(x)]
    if len(s) < 2:
        return np.nan
    return float(np.mean([len(a & b) / len(a | b) for i, a in enumerate(s) for b in s[i + 1:]]))


summ = []
for mname, g in df.groupby("method"):
    per_rep = g.groupby("rep")["cindex"].mean()
    sets = [o["sets"][mname] for o in out if mname in o["sets"]]
    summ.append(dict(method=mname, mean_C=per_rep.mean(), sd_over_repeats=per_rep.std(),
                     ci_low=np.percentile(per_rep, 2.5), ci_high=np.percentile(per_rep, 97.5),
                     fold_sd=g["cindex"].std(),
                     mean_n_genes=np.mean([len(x) for x in sets]) if sets else np.nan,
                     jaccard_stability=jaccard_stability(sets) if sets else np.nan))
if (R / "alasso_noanchor_tcga.json").exists():  # anchoring control (same splits)
    na = pd.read_json(R / "alasso_noanchor_tcga.json")
    pr = na.groupby("rep").cindex.mean()
    summ.append(dict(method="Adaptive LASSO (no anchoring)", mean_C=pr.mean(), sd_over_repeats=pr.std(),
                     ci_low=np.percentile(pr, 2.5), ci_high=np.percentile(pr, 97.5), fold_sd=na.cindex.std(),
                     mean_n_genes=na.n_genes.mean(), jaccard_stability=jaccard_stability(list(na.genes))))
summ = pd.DataFrame(summ).sort_values("mean_C", ascending=False)
summ.to_csv(R / "cv_summary_all.csv", index=False)
print(summ.round(3).to_string(index=False), "\n")
n = 79
n_test = n / 5
n_train = n - n_test

# ---------- corrected resampled t-test (Nadeau & Bengio 2003), reference vs each
piv = df.pivot_table(index=["rep", "fold"], columns="method", values="cindex")
J = len(piv)
tests = []
for m in piv.columns:
    if m == ref:
        continue
    d = (piv[ref] - piv[m]).values
    var = d.var(ddof=1) * (1 / J + n_test / n_train)
    tstat = d.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(tstat), J - 1)
    wins = (d > 0).mean()
    tests.append(dict(comparison=f"{ref} vs {m}", mean_diff=d.mean(), t=tstat, p_corrected=p, win_rate=wins))
tests = pd.DataFrame(tests).sort_values("mean_diff")
tests.to_csv(R / "cv_tests.csv", index=False)
print(tests.round(4).to_string(index=False))

# ---------- Figure 2: CV C-index with 95% interval over repeats
s = summ.sort_values("mean_C")
fig, ax = plt.subplots(figsize=(6.2, 3.6))
y = np.arange(len(s))
for i, (_, r) in enumerate(s.iterrows()):
    col = BLUE if r.method in (ref, "Anchor-weighted adaptive LASSO") else (ORANGE if r.method.startswith("OASIS") else GRAY)
    ax.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=2, solid_capstyle="round")
    ax.scatter(r.mean_C, i, s=36, color=col, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.text(r.ci_high + 0.006, i, f"{r.mean_C:.3f}", va="center", fontsize=8, color=INK2)
LBL = {"Anchor-weighted adaptive LASSO": "Anchored adaptive LASSO",
       "Clinical markers (chr3 loss, 8q gain, BAP1)": "Molecular markers (chr3, 8q, BAP1)",
       "Adaptive LASSO (no anchoring)": "Adaptive LASSO, no anchoring", "OASIS (full)": "OASIS agent (original)"}
ax.set_yticks(y, [LBL.get(m, m) for m in s.method])
ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
ax.set_xlabel("C-index (10 × 5-fold CV)")
ax.set_xlim(max(0.45, s.ci_low.min() - 0.03), min(1.0, s.ci_high.max() + 0.06))
ax.grid(axis="x", color="#e6e5e0", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(F / "figS1_tcga_cv_detail.png")
plt.close(fig)

# ---------- Figure 3: RL learning curve (mean reward per iteration, over folds)
H = np.array([o["history"] for o in out])  # OASIS (full)
mu, lo, hi = H.mean(0), np.percentile(H, 10, 0), np.percentile(H, 90, 0)
fig, ax = plt.subplots(figsize=(4.2, 2.8))
it = np.arange(1, H.shape[1] + 1)
ax.fill_between(it, lo, hi, color=BLUE, alpha=0.18, lw=0)
ax.plot(it, mu, color=BLUE, lw=2)
ax.set_xlabel("Policy iteration")
ax.set_ylabel("Mean batch reward")
ax.grid(color="#e6e5e0", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(F / "figS2_learning_curve.png")
plt.close(fig)

# ---------- Figure 4: compactness vs stability
SHORT = {"Univariate top-10 + ridge Cox": "Univariate top-10", "Anchor-weighted adaptive LASSO": "Anchored adaptive LASSO",
         "Adaptive LASSO (no anchoring)": "Adaptive LASSO, no anchoring",
         "Cox elastic net": "Elastic net", "Cox-LASSO": "Cox-LASSO", "OASIS (full)": "OASIS",
         "OASIS w/o anchoring prior": "OASIS, no anchor prior", "OASIS with random prior": "OASIS, random prior"}
sel = summ[summ.method.isin(SHORT)].dropna(subset=["mean_n_genes"])
fig, ax = plt.subplots(figsize=(5.0, 3.3))
for _, r in sel.iterrows():
    col = BLUE if r.method == "Anchor-weighted adaptive LASSO" else (ORANGE if r.method.startswith("OASIS") else GRAY)
    ax.scatter(r.mean_n_genes, r.jaccard_stability, s=48, color=col, edgecolor=SURF, linewidth=1.5, zorder=3)
    dx, ha = (6, "left") if r.mean_n_genes < 30 else (-6, "right")
    ax.annotate(SHORT[r.method], (r.mean_n_genes, r.jaccard_stability), xytext=(dx, 4),
                textcoords="offset points", fontsize=7.5, color=INK2, ha=ha)
ax.set_xscale("log")
ax.set_xticks([3, 5, 10, 20, 50], ["3", "5", "10", "20", "50"])
ax.minorticks_off()
ax.set_xlim(2.5, 80)
ax.set_xlabel("Mean signature size (genes, log scale)")
ax.set_ylabel("Selection stability (Jaccard)")
ax.grid(color="#e6e5e0", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(F / "figS3_jaccard_stability.png")  # main Fig 3 (Phi) is drawn by stability.py
plt.close(fig)

# ---------- gene frequency for OASIS across 50 folds
from collections import Counter
cnt = Counter(g for o in out for g in o["sets"][ref])
freq = pd.Series(cnt).sort_values(ascending=False) / len(out)
freq.to_csv(R / "rl_gene_frequency_cv.csv", header=["fraction_of_folds"])
print("\nTop OASIS genes across folds:\n", freq.head(15).round(2).to_string())
