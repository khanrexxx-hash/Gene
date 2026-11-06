"""Summaries for the additional control analyses (review2.py): TCGA paired tests (Nadeau-Bengio corrected),
sensitivity table, permuted / rescaled controls, markers + score, stratified pooled and LOCO.
Usage: python review2_summary.py RESULTS_DIR
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
R = Path(sys.argv[1])


def load(fname):
    p = R / fname
    if not p.exists():
        return None
    d = json.load(open(p))
    return pd.DataFrame([dict(rep=x.get("rep", x.get("seed")), fold=x.get("fold", x.get("split")), **x["cindex"]) for x in d])


def nb_test(diff, n_tr_over_te=0.25):
    J = len(diff); m = diff.mean(); v = diff.var(ddof=1)
    se = np.sqrt(v * (1 / J + n_tr_over_te))
    tstat = m / se if se > 0 else 0.0
    return m, 2 * stats.t.sf(abs(tstat), J - 1), (diff > 0).mean()


# ---- TCGA design
a = load("cv_results.json"); b = load("review_tcga.json"); c = load("review2_tcga.json")
a2 = json.load(open(R / "alasso_noanchor_tcga.json"))
na = pd.DataFrame([dict(rep=x["rep"], fold=x["fold"], **{"Adaptive LASSO, no anchoring": x["cindex"]}) if not isinstance(x["cindex"], dict)
                   else dict(rep=x["rep"], fold=x["fold"], **x["cindex"]) for x in a2])
T = a.merge(b, on=["rep", "fold"]).merge(c, on=["rep", "fold"]).merge(na, on=["rep", "fold"])
ref = "anchored"
print("check reproduction: anchored (review2) vs anchor-weighted (original):",
      round(T[ref].mean(), 4), round(T["Anchor-weighted adaptive LASSO"].mean(), 4))
rows = []
for m in [x for x in T.columns if x not in ("rep", "fold", ref, "Anchor-weighted adaptive LASSO", "Clinical markers (chr3 loss, 8q gain, BAP1)")]:
    md, p, w = nb_test((T[ref] - T[m]).values)
    rows.append(dict(method=m, mean_C=T[m].mean(), sd=T[m].std(), diff_anchored_minus=md, p_corrected=p, anchored_wins=w))
tc = pd.DataFrame(rows).sort_values("mean_C", ascending=False)
tc.loc[len(tc)] = dict(method="anchored (reference)", mean_C=T[ref].mean(), sd=T[ref].std())
tc.to_csv(R / "review2_tcga_summary.csv", index=False)
print(tc.round(3).to_string(index=False))
# markers + score vs markers alone
md, p, w = nb_test((T["markers + anchored score"] - T["markers alone (refit)"]).values)
print("markers+score minus markers alone:", round(md, 3), "p", round(p, 3), "wins", round(w, 2))
md, p, w = nb_test((T["markers + anchored score"] - T[ref]).values)
print("markers+score minus anchored alone:", round(md, 3), "p", round(p, 3), "wins", round(w, 2))

# ---- pooled and LOCO
for fn in ["review2_pooled.json", "review2_loco.json"]:
    d = load(fn)
    if d is None:
        continue
    print("\n", fn, d.shape)
    print(d.drop(columns=["rep", "fold"]).mean().round(3).to_string())
    d.to_csv(R / fn.replace(".json", "_folds.csv"), index=False)

# ---- pooled paired tests (same 50 splits as the main pooled analysis)
import glob
base = {}
for f in glob.glob(str(R / "pooled_cv" / "*.json")):
    o = json.load(open(f))
    base[o["split"]] = o["cindex"]
na = pd.read_csv(R / "alasso_noanchor_pooled.csv").set_index("split").cindex
rv = json.load(open(R / "review2_pooled.json"))
rows = []
P_ = pd.DataFrame([dict(split=o["split"], anchored=base[o["split"]]["Anchor-weighted adaptive LASSO"],
                        **{"Adaptive LASSO, no anchoring": na[o["split"]]}, **o["cindex"]) for o in rv])
pairs = [("anchored", m) for m in ["Adaptive LASSO, no anchoring", "permuted anchoring (mean of 10)", "rescaled prior",
                                    "positional prior + 6p"]] + \
        [("stratified: anchored", "stratified: no anchoring"), ("stratified: anchored", "stratified: positional prior"),
         ("rescaled prior", "Adaptive LASSO, no anchoring"), ("permuted anchoring (mean of 10)", "Adaptive LASSO, no anchoring")]
for a_, b_ in pairs:
    d = (P_[a_] - P_[b_]).values
    J = len(d); se = np.sqrt(d.var(ddof=1) * (1 / J + 0.25))
    p = 2 * stats.t.sf(abs(d.mean() / se), J - 1)
    mdd = (stats.t.ppf(0.975, J - 1) + stats.t.ppf(0.8, J - 1)) * se
    rows.append(dict(a=a_, b=b_, mean_a=P_[a_].mean(), mean_b=P_[b_].mean(), diff=d.mean(), p_corrected=p,
                     a_wins=(d > 0).mean(), mdd=mdd))
pp = pd.DataFrame(rows)
pp.to_csv(R / "review2_pooled_paired.csv", index=False)
print(pp.round(3).to_string(index=False))
