"""Compare OASIS-2 variants with all v1 methods on the same folds.

Usage: python compare_v2.py RESULTS_DIR REP_START REP_END
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

R = Path(sys.argv[1]); R0, R1 = int(sys.argv[2]), int(sys.argv[3])
v1 = {(o["rep"], o["fold"]): o for o in json.load(open(R / "cv_results.json"))}
rows = []
for f in glob.glob(str(R / "v2_folds" / "*.json")):
    o = json.load(open(f))
    if not (R0 <= o["rep"] <= R1):
        continue
    rows.append(dict(rep=o["rep"], fold=o["fold"], method=f"OASIS-2 {o['var']} (single)", cindex=o["c_single"],
                     n_genes=len(o["sel"]), sel=o["sel"]))
    rows.append(dict(rep=o["rep"], fold=o["fold"], method=f"OASIS-2 {o['var']} (ensemble)", cindex=o["c_ens"],
                     n_genes=o["n_ens_genes"], sel=o["sel"]))
for (rep, k), o in v1.items():
    if R0 <= rep <= R1:
        for m, c in o["cindex"].items():
            rows.append(dict(rep=rep, fold=k, method=m, cindex=c, n_genes=len(o["sets"].get(m, [])) or np.nan,
                             sel=o["sets"].get(m, [])))
df = pd.DataFrame(rows)


def jac(sets):
    s = [set(x) for x in sets if len(x)]
    return np.mean([len(a & b) / len(a | b) for i, a in enumerate(s) for b in s[i + 1:]]) if len(s) > 1 else np.nan


piv = df.pivot_table(index=["rep", "fold"], columns="method", values="cindex")
J, corr = len(piv), 1 / len(piv) + 0.25
summ = []
for m, g in df.groupby("method"):
    summ.append(dict(method=m, mean_C=g.groupby("rep").cindex.mean().mean(), folds=len(g),
                     genes=g.n_genes.mean(), stability=jac(g.sel)))
summ = pd.DataFrame(summ).sort_values("mean_C", ascending=False)
print(summ.round(3).to_string(index=False))
best = [m for m in summ.method if m.startswith("OASIS-2")][0]
print(f"\nPaired tests vs {best} (corrected resampled t-test):")
for m in ["Random survival forest", "Univariate top-10 + ridge Cox", "Cox elastic net", "Cox-LASSO", "OASIS (full)",
          "Clinical markers (chr3 loss, 8q gain, BAP1)", "Anchor-weighted adaptive LASSO"]:
    if m in piv:
        d = (piv[best] - piv[m]).dropna().values
        t = d.mean() / np.sqrt(d.var(ddof=1) * corr)
        print(f"  vs {m:45s} diff {d.mean():+.3f}  p={2 * stats.t.sf(abs(t), len(d) - 1):.3f}  wins {np.mean(d > 0):.2f}")
summ.to_csv(R / f"v2_compare_reps{R0}-{R1}.csv", index=False)
