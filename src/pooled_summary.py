"""Summarise pooled CV and leave-one-cohort-out results.

Usage: python pooled_summary.py RESULTS_DIR [reference method]
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

R = Path(sys.argv[1])
REF = sys.argv[2] if len(sys.argv) > 2 else "OASIS-2 (policy ensemble)"

# ---------- pooled CV
cv = [json.load(open(f)) for f in glob.glob(str(R / "pooled_cv" / "*.json"))]
if cv:
    rows = []
    for o in cv:
        rep = int(o["split"].split("_")[0][1:])
        for m, c in o["cindex"].items():
            rows.append(dict(split=o["split"], rep=rep, method=m, cindex=c, n_genes=o["n_genes"][m]))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="split", columns="method", values="cindex")
    J = len(piv); corr = 1 / J + 0.25
    summ = df.groupby("method").agg(mean_C=("cindex", "mean"), genes=("n_genes", "mean")).reset_index()
    per_rep = df.groupby(["method", "rep"]).cindex.mean().groupby("method")
    summ["ci_low"] = summ.method.map(per_rep.quantile(0.025)); summ["ci_high"] = summ.method.map(per_rep.quantile(0.975))
    tests = []
    for m in piv.columns:
        if m == REF:
            continue
        d = (piv[REF] - piv[m]).values
        se = np.sqrt(d.var(ddof=1) * corr)
        t = d.mean() / se
        tests.append(dict(method=m, diff_vs_ref=d.mean(), p_corrected=2 * stats.t.sf(abs(t), J - 1),
                          win_rate=(d > 0).mean(),
                          min_detectable=(stats.t.ppf(0.975, J - 1) + stats.t.ppf(0.8, J - 1)) * se))
    summ = summ.merge(pd.DataFrame(tests), on="method", how="left").sort_values("mean_C", ascending=False)
    summ.to_csv(R / "pooled_cv_summary.csv", index=False)
    print(f"POOLED CV ({J} folds, ~160 train / ~40 test, within-cohort C)  reference = {REF}")
    print(summ.round(3).to_string(index=False))

# ---------- leave-one-cohort-out
lo = [json.load(open(f)) for f in glob.glob(str(R / "pooled_loco" / "*.json"))]
if lo:
    rows = []
    for o in lo:
        coh = o["split"].split("_")[1]
        for m, c in o["cindex"].items():
            rows.append(dict(test=coh, method=m, cindex=c, n_genes=o["n_genes"][m], anchored=o["anchored"]))
    d = pd.DataFrame(rows)
    tab = d.pivot_table(index="method", columns="test", values="cindex", aggfunc="mean")
    tab["mean"] = tab.mean(1)
    tab = tab.sort_values("mean", ascending=False)
    tab.to_csv(R / "pooled_loco_summary.csv")
    print("\nLEAVE-ONE-COHORT-OUT (mean of 5 seeds; TCGA column = no anchoring available)")
    print(tab.round(3).to_string())
