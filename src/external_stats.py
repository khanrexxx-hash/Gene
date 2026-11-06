"""Fixed-model external evaluation statistics.

Harrell's C with bootstrap CI and paired bootstrap tests, Uno's C, time-dependent AUC at
36 and 60 months, Brier score at 36 months (TCGA-recalibrated), Kaplan-Meier curves with
numbers at risk, and monosomy-3-adjusted Cox models for every method.

Usage: python external_stats.py EXT_DIR GEO_DIR FIG_DIR [REF]
"""
import sys, gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, brier_score
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).parent))
from rl_agent import cindex

X, G, F = map(Path, sys.argv[1:4])
REF = sys.argv[4] if len(sys.argv) > 4 else "Anchored adaptive LASSO"
F.mkdir(parents=True, exist_ok=True)
rs = pd.read_csv(X / "external_risk_scores.csv")
rng = np.random.default_rng(42)
B = 1000
BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "savefig.dpi": 300})

# ---------- discrimination, calibration-type metrics, paired tests
rows = []
for coh, g in rs.groupby("cohort"):
    piv = g.pivot(index="sample", columns="method", values="risk")
    y = g.drop_duplicates("sample").set_index("sample").loc[piv.index]
    t, e = y["time"].values, y["event"].values
    ys = Surv.from_arrays(e.astype(bool), t)
    tau = min(60.0, np.percentile(t[e == 1], 90))
    n = len(t)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    boot = {m: np.array([cindex(piv[m].values[i], t[i], e[i]) for i in idx]) for m in piv.columns}
    for m in piv.columns:
        r = piv[m].values
        c = cindex(r, t, e)
        d = boot[REF] - boot[m]
        uno = concordance_index_ipcw(ys, ys, r, tau=tau)[0]
        aucs = {}
        for h in (36, 60):
            try:
                aucs[h] = cumulative_dynamic_auc(ys, ys, r, [h])[0][0]
            except ValueError:
                aucs[h] = np.nan
        s36 = g[g.method == m].set_index("sample").loc[piv.index, "surv36"].values
        try:
            # not estimable when the TCGA recalibration diverges (training score separates TCGA outcomes)
            bs = brier_score(ys, ys, s36[:, None], [36])[1][0] if np.isfinite(s36).all() and np.std(s36) > 1e-6 else np.nan
        except ValueError:
            bs = np.nan
        rows.append(dict(cohort=coh, method=m, harrell_C=c, ci_low=np.percentile(boot[m], 2.5),
                         ci_high=np.percentile(boot[m], 97.5), uno_C=uno, uno_tau=tau,
                         tdAUC_36=aucs[36], tdAUC_60=aucs[60], brier_36=bs,
                         diff_vs_ref=cindex(piv[REF].values, t, e) - c,
                         p_boot=np.nan if m == REF else 2 * min((d <= 0).mean(), (d >= 0).mean())))
ext = pd.DataFrame(rows)
ext.to_csv(X / "external_metrics.csv", index=False)
print(ext.round(3).to_string(index=False))


# ---------- Kaplan-Meier with numbers at risk (cut-off = cohort median, not tuned)
def km_panel(ax, axr, t, e, r, title):
    hi = r > np.median(r)
    ticks = np.arange(0, 121, 24)
    for k, (mask, lab, col) in enumerate([(~hi, "Low risk", BLUE), (hi, "High risk", ORANGE)]):
        x, s = kaplan_meier_estimator(e[mask].astype(bool), t[mask])
        ax.step(np.r_[0, x], np.r_[1, s], where="post", color=col, lw=2, label=f"{lab} (n={mask.sum()})")
        axr.text(-6, 1 - k, lab, ha="right", va="center", fontsize=7.5, color=col)
        for tk in ticks:
            axr.text(tk, 1 - k, str(int((t[mask] >= tk).sum())), ha="center", va="center", fontsize=7.5, color=INK2)
    chi, p = compare_survival(Surv.from_arrays(e.astype(bool), t), hi.astype(int))
    ax.set_title(title, fontsize=9, loc="left")
    ax.text(0.97, 0.93, f"log-rank p = {p:.2g}", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.03); ax.set_xlim(-2, 122); ax.set_xticks(ticks)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    axr.set_xlim(-2, 122); axr.set_ylim(-0.6, 1.6); axr.axis("off")
    axr.set_title("Number at risk", fontsize=7.5, loc="left", color=INK2)
    return p


fig = plt.figure(figsize=(7.4, 3.6))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.55, wspace=0.28)
kmp = {}
for j, (coh, g) in enumerate(rs[rs.method == REF].groupby("cohort")):
    ax = fig.add_subplot(gs[0, j]); axr = fig.add_subplot(gs[1, j])
    kmp[coh] = km_panel(ax, axr, g.time.values, g.event.values, g.risk.values, f"{coh}: {REF}")
    ax.set_xlabel("Months", fontsize=8)
    if True:
        ax.set_ylabel("Metastasis-free probability" if coh == "GSE22138" else "Metastatic-death-free probability")
fig.savefig(F / "fig4_km_external.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
pd.Series(kmp).to_csv(X / "km_logrank.csv", header=["p"])
print("log-rank:", kmp)


# ---------- monosomy-3-adjusted Cox for every method (GSE22138)
def cox_wald(Xm, t, e, iters=50):
    o = np.argsort(-t, kind="mergesort")
    Xm, e = Xm[o], e[o]
    beta = np.zeros(Xm.shape[1])
    for _ in range(iters):
        w = np.exp(Xm @ beta)
        cw = np.cumsum(w)
        mu = np.cumsum(w[:, None] * Xm, 0) / cw[:, None]
        cxx = np.cumsum(w[:, None, None] * Xm[:, :, None] * Xm[:, None, :], 0)
        gr = (e[:, None] * (Xm - mu)).sum(0)
        H = sum(cxx[i] / cw[i] - np.outer(mu[i], mu[i]) for i in np.where(e == 1)[0])
        step = np.linalg.solve(H, gr)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se, 2 * stats.norm.sf(np.abs(beta / se))


lines = gzip.open(G / "GSE22138.txt.gz", "rt").read().splitlines()
gsm = [x.strip('"') for x in next(l for l in lines if l.startswith("!Sample_geo_accession")).split("\t")[1:]]
chr3 = [x.strip('"').split(": ")[1] for x in next(l for l in lines if "chromosome 3 status" in l).split("\t")[1:]]
c3 = pd.Series(chr3, index=gsm)
out = []
for mname, gm in rs[rs.cohort == "GSE22138"].groupby("method"):
    gm = gm.set_index("sample").join(c3.rename("chr3"))
    gm = gm[gm.chr3 != "NA"]
    if gm.risk.std() == 0:
        continue
    gm["mono3"] = (gm.chr3 != "disomy").astype(float)
    gm["risk_z"] = (gm.risk - gm.risk.mean()) / gm.risk.std()
    b, se, p = cox_wald(gm[["risk_z", "mono3"]].values, gm.time.values, gm.event.values)
    sub = gm[gm.mono3 == 1]
    out.append(dict(method=mname, n=len(gm), events=int(gm.event.sum()),
                    C=cindex(gm.risk.values, gm.time.values, gm.event.values),
                    C_mono3_alone=cindex(gm.mono3.values, gm.time.values, gm.event.values),
                    HR_per_SD=np.exp(b[0]), HR_lo=np.exp(b[0] - 1.96 * se[0]), HR_hi=np.exp(b[0] + 1.96 * se[0]),
                    p_score_adj=p[0], HR_mono3=np.exp(b[1]), p_mono3_adj=p[1], n_mono3=len(sub),
                    C_within_mono3=cindex(sub.risk.values, sub.time.values, sub.event.values)))
out = pd.DataFrame(out).sort_values("p_score_adj")
out.to_csv(X / "chr3_adjusted_all_methods.csv", index=False)
print(out.round(3).to_string(index=False))
