"""Repeated stratified 5-fold CV on TCGA-UVM (PFS) for OASIS, ablations and baselines.

Usage: python run_cv.py PROCESSED_DIR OUT_DIR [n_repeats]
Writes fold-level C-indices and selected gene sets to OUT_DIR.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).parent))
from rl_agent import OASIS, ridge_cox, cindex, univariate_cox_z, anchoring_score

warnings.filterwarnings("ignore")
P, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
OUT.mkdir(parents=True, exist_ok=True)

E = pd.read_csv(P / "expr.csv.gz", index_col=0)
CNA = pd.read_csv(P / "cna.csv.gz", index_col=0)
METH = pd.read_csv(P / "meth.csv.gz", index_col=0)
CL = pd.read_csv(P / "clinical.csv", index_col=0)
T, EV = CL["PFS_months"].values, CL["PFS_event"].values
CLIN_COLS = ["chr3_loss", "chr8q_gain", "BAP1_mut"]  # age/stage are empty in this release


def z(train, test):
    m, s = train.mean(0), train.std(0) + 1e-8
    return (train - m) / s, (test - m) / s


def coxnet_cv(Xtr, ttr, etr, l1_ratio, penalty_factor=None, seed=0):
    """Coxnet with alpha chosen by inner 3-fold CV (C-index)."""
    y = Surv.from_arrays(etr.astype(bool), ttr)
    kw = dict(l1_ratio=l1_ratio, alpha_min_ratio=0.05, n_alphas=30, max_iter=20000)
    if penalty_factor is not None:
        kw["penalty_factor"] = penalty_factor
    full = CoxnetSurvivalAnalysis(**kw, fit_baseline_model=False).fit(Xtr, y)
    alphas = full.alphas_
    scores = np.zeros(len(alphas))
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    for a, b in inner.split(Xtr, etr):
        try:
            m = CoxnetSurvivalAnalysis(**{**kw, "alphas": alphas}).fit(Xtr[a], y[a])
        except Exception:
            continue
        for j in range(len(alphas)):
            coef = m.coef_[:, j] if j < m.coef_.shape[1] else m.coef_[:, -1]
            r = Xtr[b] @ coef
            scores[j] += cindex(r, ttr[b], etr[b]) if np.any(coef) else 0.5
    j = int(np.argmax(scores))
    coef = full.coef_[:, j]
    return coef


def run_fold(rep, k, tr, te):
    res, sets = {}, {}
    Etr, Ete = E.iloc[tr], E.iloc[te]
    ttr, etr, tte, ete = T[tr], EV[tr], T[te], EV[te]
    Xtr, Xte = z(Etr.values, Ete.values)
    genes = np.asarray(E.columns)

    # --- clinical reference (chr3 loss, 8q gain, BAP1, age)
    Ctr, Cte = z(CL.iloc[tr][CLIN_COLS].values.astype(float), CL.iloc[te][CLIN_COLS].values.astype(float))
    b = ridge_cox(Ctr, ttr, etr, 1.0)
    res["Clinical markers (chr3 loss, 8q gain, BAP1)"] = cindex(Cte @ b, tte, ete)

    # --- univariate screen + ridge Cox (top 10)
    zs = np.abs(univariate_cox_z(Xtr, ttr, etr))
    top = np.argsort(-zs)[:10]
    b = ridge_cox(Xtr[:, top], ttr, etr, 1.0)
    res["Univariate top-10 + ridge Cox"] = cindex(Xte[:, top] @ b, tte, ete)
    sets["Univariate top-10 + ridge Cox"] = list(genes[top])

    # --- Cox-LASSO / elastic net on all genes
    for name, l1 in [("Cox-LASSO", 1.0), ("Cox elastic net", 0.5)]:
        coef = coxnet_cv(Xtr, ttr, etr, l1, seed=rep)
        res[name] = cindex(Xte @ coef, tte, ete) if np.any(coef) else 0.5
        sets[name] = list(genes[coef != 0])

    # --- anchor-weighted adaptive LASSO (same prior as OASIS, no RL)
    cand = np.argsort(-zs)[:300]
    a = anchoring_score(Etr.iloc[:, cand], CNA.iloc[tr], METH.iloc[tr]).values
    pr = (zs[cand] - zs[cand].mean()) / zs[cand].std() + (a - a.mean()) / (a.std() + 1e-8)
    pf = np.exp(-pr / 2); pf = pf / pf.mean()
    coef = coxnet_cv(Xtr[:, cand], ttr, etr, 1.0, penalty_factor=pf, seed=rep)
    res["Anchor-weighted adaptive LASSO"] = cindex(Xte[:, cand] @ coef, tte, ete) if np.any(coef) else 0.5
    sets["Anchor-weighted adaptive LASSO"] = list(genes[cand][coef != 0])

    # --- RSF on top-500 screened genes
    c5 = np.argsort(-zs)[:500]
    rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                               n_jobs=1, random_state=rep).fit(Xtr[:, c5], Surv.from_arrays(etr.astype(bool), ttr))
    res["Random survival forest"] = cindex(rsf.predict(Xte[:, c5]), tte, ete)

    # --- OASIS and ablations
    variants = {
        "OASIS (full)": dict(),
        "OASIS w/o anchoring prior": dict(use_anchor=False),
        "OASIS w/o stability reward": dict(use_stability=False),
        "OASIS with random prior": dict(random_prior=True),
    }
    for name, kw in variants.items():
        m = OASIS(seed=1000 * rep + k, **kw).fit(Etr, ttr, etr, CNA.iloc[tr], METH.iloc[tr])
        res[name] = cindex(m.predict_risk(Ete), tte, ete)
        sets[name] = m.selected_
        if name == "OASIS (full)":
            hist = m.history_
    return dict(rep=rep, fold=k, cindex=res, sets=sets, history=hist)


folds = []
for rep in range(REPEATS):
    skf = StratifiedKFold(5, shuffle=True, random_state=rep)
    for k, (tr, te) in enumerate(skf.split(E.values, EV)):
        folds.append((rep, k, tr, te))

out = Parallel(n_jobs=2, verbose=5)(delayed(run_fold)(*f) for f in folds)
json.dump(out, open(OUT / "cv_results.json", "w"))

# --- summary
rows = []
for o in out:
    for mname, c in o["cindex"].items():
        rows.append(dict(rep=o["rep"], fold=o["fold"], method=mname, cindex=c))
df = pd.DataFrame(rows)
df.to_csv(OUT / "cv_fold_cindex.csv", index=False)


def jaccard_stability(set_list):
    s = [set(x) for x in set_list if len(x)]
    if len(s) < 2:
        return np.nan
    v = [len(a & b) / len(a | b) for i, a in enumerate(s) for b in s[i + 1:]]
    return float(np.mean(v))


summ = []
for mname, g in df.groupby("method"):
    per_rep = g.groupby("rep")["cindex"].mean()
    sets = [o["sets"].get(mname, []) for o in out]
    summ.append(dict(method=mname, mean_C=per_rep.mean(), sd_over_repeats=per_rep.std(),
                     ci_low=np.percentile(per_rep, 2.5), ci_high=np.percentile(per_rep, 97.5),
                     mean_n_genes=np.mean([len(x) for x in sets]) if any(sets) else np.nan,
                     jaccard_stability=jaccard_stability(sets)))
summ = pd.DataFrame(summ).sort_values("mean_C", ascending=False)
summ.to_csv(OUT / "cv_summary.csv", index=False)
print(summ.round(3).to_string(index=False))
