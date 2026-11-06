"""Additional fixed-model external analyses.

1. Calibration at 36 months for each fixed model (scores recalibrated on TCGA only):
   calibration slope (Cox regression of the outcome on the TCGA-calibrated linear predictor),
   calibration-in-the-large (Kaplan-Meier observed 36-month risk minus mean predicted risk),
   IPCW Brier score, Brier score of a Kaplan-Meier null model, and scaled Brier (IPA).
   Supplementary figure: observed vs predicted 36-month risk by risk tertile.
2. GSE22138 sensitivity analysis excluding metastases diagnosed within one month.
3. Events within the monosomy-3 subgroup of GSE22138.
4. Molecular marker + score in GSE22138: Cox model [monosomy 3, anchored score] fitted on TCGA
   (chr3 loss + score), applied unchanged; compared with monosomy 3 alone (paired bootstrap).
5. Single-patient scoring sheet for the anchored model: TCGA reference mean/SD per gene,
   coefficients, and baseline survival S0(36) at the TCGA mean score.

Usage: python external_extra.py PROCESSED_DIR EXT_DIR GEO_DIR FIG_DIR
"""
import sys, gzip, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sksurv.metrics import brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).parent))
from rl_agent import cindex
from external_stats_helpers import cox_fit, km_at

P, X, G, F = map(Path, sys.argv[1:5])
rs = pd.read_csv(X / "external_risk_scores.csv")
MAIN = ["Anchored adaptive LASSO", "Adaptive LASSO, no anchoring", "Adaptive LASSO, positional prior",
        "Cox-LASSO", "Cox elastic net", "Univariate top-10 + ridge Cox", "15-gene-class proxy",
        "15-gene-class proxy + PRAME", "Random survival forest", "OASIS-2 agent (policy ensemble)"]
H = 36.0

# ---------- 1. calibration
rows, tert = [], []
for coh, g in rs.groupby("cohort"):
    for m in MAIN:
        d = g[g.method == m]
        t, e, s = d.time.values, d.event.values, np.clip(d.surv36.values, 1e-6, 1 - 1e-6)
        ys = Surv.from_arrays(e.astype(bool), t)
        if np.std(s) < 1e-6:   # TCGA recalibration diverged: training score separates TCGA outcomes
            rows.append(dict(cohort=coh, method=m, note="not estimable: apparent TCGA C ~ 1"))
            continue
        lp = np.log(-np.log(s))                       # TCGA-calibrated linear predictor (+ constant)
        b, se = cox_fit(lp[:, None], t, e)
        obs = 1 - km_at(t, e, H)
        bs = brier_score(ys, ys, s[:, None], [H])[1][0]
        bs0 = brier_score(ys, ys, np.full((len(t), 1), 1 - obs), [H])[1][0]
        rows.append(dict(cohort=coh, method=m, slope=b[0], slope_lo=b[0] - 1.96 * se[0], slope_hi=b[0] + 1.96 * se[0],
                         observed_risk36=obs, mean_predicted_risk36=1 - s.mean(), citl=obs - (1 - s.mean()),
                         brier36=bs, brier36_null_KM=bs0, IPA=1 - bs / bs0))
        if m == MAIN[0]:
            q = pd.qcut(s, 3, labels=False)
            for k in range(3):
                mk = q == k
                tert.append(dict(cohort=coh, tertile=k, n=mk.sum(), predicted=1 - s[mk].mean(),
                                 observed=1 - km_at(t[mk], e[mk], H)))
cal = pd.DataFrame(rows)
cal.to_csv(X / "calibration36.csv", index=False)
print(cal.round(3).to_string(index=False))
tert = pd.DataFrame(tert)
tert.to_csv(X / "calibration36_tertiles.csv", index=False)
print(tert.round(3))

fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1))
for ax, (coh, g) in zip(axs, tert.groupby("cohort")):
    ax.plot([0, 1], [0, 1], color="#b0afaa", lw=1, ls="--")
    ax.plot(g.predicted, g.observed, "o-", color="#2a78d6", lw=1.6)
    for _, r in g.iterrows():
        ax.annotate(f"n={int(r.n)}", (r.predicted, r.observed), xytext=(6, 4), textcoords="offset points", fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted 36-month risk (TCGA-recalibrated)", fontsize=8)
    ax.set_ylabel("Observed 36-month risk (Kaplan-Meier)", fontsize=8)
    ax.set_title(coh, fontsize=9, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(F / "figS4_calibration36.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 2. GSE22138 without metastases within one month
g = rs[rs.cohort == "GSE22138"]
early = g[(g.method == MAIN[0]) & (g.event == 1) & (g.time < 1)][["sample", "time"]]
print("early events:", early.values.tolist())
sens = []
for m in MAIN:
    d = g[g.method == m]
    keep = ~d["sample"].isin(early["sample"])
    sens.append(dict(method=m, C_all=cindex(d.risk.values, d.time.values, d.event.values),
                     C_excl_early=cindex(d.risk.values[keep], d.time.values[keep], d.event.values[keep]),
                     n=int(keep.sum()), events=int(d.event.values[keep].sum())))
sens = pd.DataFrame(sens)
sens.to_csv(X / "gse22138_exclude_early.csv", index=False)
print(sens.round(3).to_string(index=False))

# ---------- 3./4. monosomy 3
lines = gzip.open(G / "GSE22138.txt.gz", "rt").read().splitlines()
gsm = [x.strip('"') for x in next(l for l in lines if l.startswith("!Sample_geo_accession")).split("\t")[1:]]
chr3 = [x.strip('"').split(": ")[1] for x in next(l for l in lines if "chromosome 3 status" in l).split("\t")[1:]]
c3 = pd.Series(chr3, index=gsm)
d = g[g.method == MAIN[0]].set_index("sample").join(c3.rename("chr3"))
d = d[d.chr3 != "NA"].copy()
d["mono3"] = (d.chr3 != "disomy").astype(float)
mono = dict(n_known=len(d), n_mono3=int(d.mono3.sum()), events_mono3=int(d.event[d.mono3 == 1].sum()),
            n_disomy=int((d.mono3 == 0).sum()), events_disomy=int(d.event[d.mono3 == 0].sum()))
print(mono)

E = pd.read_csv(P / "expr.csv.gz", index_col=0)
CL = pd.read_csv(P / "clinical.csv", index_col=0).loc[E.index]
fm = json.load(open(X / "fixed_models.json"))[MAIN[0]]
genes, beta = fm["genes"], np.array(fm["beta"])
mu, sd = E[genes].mean(), E[genes].std()
lp_tcga = ((E[genes] - mu) / sd).values @ beta
t0, e0 = CL.PFS_months.values, CL.PFS_event.values
s_lp = lp_tcga.std()
bm, sem = cox_fit(np.c_[CL.chr3_loss.values, lp_tcga / s_lp], t0, e0)
comb = bm[0] * d.mono3.values + bm[1] * d.risk.values / s_lp
tt, ee = d.time.values, d.event.values
rng = np.random.default_rng(7)
bt = []
for _ in range(2000):
    i = rng.integers(0, len(tt), len(tt))
    bt.append(cindex(comb[i], tt[i], ee[i]) - cindex(d.mono3.values[i], tt[i], ee[i]))
bt = np.array(bt)
mk = dict(tcga_coef_chr3=bm[0], tcga_coef_score_perSD=bm[1], C_mono3=cindex(d.mono3.values, tt, ee),
          C_score=cindex(d.risk.values, tt, ee), C_mono3_plus_score=cindex(comb, tt, ee),
          diff=cindex(comb, tt, ee) - cindex(d.mono3.values, tt, ee),
          diff_lo=np.percentile(bt, 2.5), diff_hi=np.percentile(bt, 97.5),
          p_boot=2 * min((bt <= 0).mean(), (bt >= 0).mean()))
print(mk)

# ---------- 5. single-patient scoring sheet
# TCGA recalibration (as used for every external probability): 1-D Cox on the score, Breslow baseline at score 0
k_rec = cox_fit(lp_tcga[:, None], t0, e0)[0][0]
H0 = 0.0
for i in np.where((e0 == 1) & (t0 <= H))[0]:
    H0 += 1.0 / np.exp(k_rec * lp_tcga[t0 >= t0[i]]).sum()
sheet = pd.DataFrame({"gene": genes, "tcga_mean_log2_rsem_plus1": mu.values, "tcga_sd": sd.values, "coef_per_sd": beta})
sheet.to_csv(X / "single_patient_scoring_sheet.csv", index=False)
extra = dict(lp_mean_tcga=float(lp_tcga.mean()), recalibration_slope=float(k_rec), S0_36_at_score0=float(np.exp(-H0)),
             note="score = sum coef*(x-mean)/sd with x = log2(RSEM+1); S(36|x) = S0_36 ** exp(slope * score). "
                  "Valid only for TCGA-comparable RNA-seq; not validated for single samples on other platforms.")
print(sheet.round(3).to_string(index=False)); print(extra)
json.dump(dict(monosomy3=mono, markers_plus_score_GSE22138=mk, scoring=extra,
               early_events=early.values.tolist()), open(X / "external_extra.json", "w"), indent=1, default=float)
