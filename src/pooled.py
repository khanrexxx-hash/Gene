"""Pooled uveal melanoma analysis: TCGA-UVM + GSE22138 + GSE44295 (199 patients).

Usage: python pooled.py PROCESSED_DIR GEO_DIR OUT_DIR MODE [rep_start rep_end]
  MODE = cv    : 10x5-fold CV on the pooled cohort, folds stratified by cohort x event
  MODE = loco  : leave-one-cohort-out (train on two cohorts, test on the third)

Each cohort is z-scored per gene within cohort over the genes shared by all platforms.
Test C-indices are computed within cohort (pairs across cohorts are never compared).
The anchoring prior uses copy number / methylation of the TCGA patients in the TRAINING
set only; when TCGA is held out (LOCO), no anchoring information is available and the
anchored methods fall back to their un-anchored versions.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).parent))
from rl_agent import OASIS, ridge_cox, cindex, univariate_cox_z, anchoring_score
from rl_agent2 import OASIS2
from run_cv_helpers import coxnet_cv
from geo_utils import read_annot, read_series, to_genes, zscore

warnings.filterwarnings("ignore")
P, G, OUT = map(Path, sys.argv[1:4])
MODE = sys.argv[4]
R0, R1 = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 9)
CK = OUT / f"pooled_{MODE}"; CK.mkdir(parents=True, exist_ok=True)

# ---------------- assemble pooled cohort
E = pd.read_csv(P / "expr.csv.gz", index_col=0)
CNA = pd.read_csv(P / "cna.csv.gz", index_col=0)
METH = pd.read_csv(P / "meth.csv.gz", index_col=0)
CL = pd.read_csv(P / "clinical.csv", index_col=0)

X1, M1 = read_series(G / "GSE22138.txt.gz"); X1 = to_genes(X1, read_annot(G / "GPL570.annot.gz"))
X2, M2 = read_series(G / "GSE44295.txt.gz"); X2 = to_genes(X2, read_annot(G / "GPL6883.annot.gz"))
y1 = pd.DataFrame({"time": pd.to_numeric(M1["months to endpoint"], errors="coerce"),
                   "event": (M1["metastasis"] == "yes").astype(int)}, index=M1.index).dropna()
keep = (M2["cell type"] == "primary uveal melanoma cells") & (M2["survival (days)"] != "unknown") & M2["metastasis"].isin(["0", "1"])
M2 = M2[keep]
y2 = pd.DataFrame({"time": pd.to_numeric(M2["survival (days)"], errors="coerce") / 30.44,
                   "event": (M2["metastasis"] == "1").astype(int)}, index=M2.index).dropna()
universe = E.columns.intersection(X1.columns).intersection(X2.columns)

parts = [(zscore(E[universe]), CL["PFS_months"], CL["PFS_event"], "TCGA"),
         (zscore(X1.loc[y1.index, universe]), y1["time"], y1["event"], "GSE22138"),
         (zscore(X2.loc[y2.index, universe]), y2["time"], y2["event"], "GSE44295")]
X = pd.concat([p[0] for p in parts]).fillna(0.0)
T = np.concatenate([p[1].values for p in parts]).astype(float)
EV = np.concatenate([p[2].values for p in parts]).astype(int)
COH = np.concatenate([[p[3]] * len(p[0]) for p in parts])
T[T <= 0] = 0.1
TCGA_IDS = set(E.index)
if __name__ == "__main__" and MODE == "info":
    print(X.shape, {c: (int((COH == c).sum()), int(EV[COH == c].sum())) for c in np.unique(COH)})
    sys.exit()


def cindex_strat(r, t, e, coh):
    num = den = 0.0
    for c in np.unique(coh):
        m = coh == c
        rr, tt, ee = r[m], t[m], e[m]
        for i in np.where(ee == 1)[0]:
            comp = tt > tt[i]
            num += (rr[i] > rr[comp]).sum() + 0.5 * (rr[i] == rr[comp]).sum()
            den += comp.sum()
    return num / den if den else 0.5


def anchor_from_training(tr_ids):
    ids = [i for i in tr_ids if i in TCGA_IDS]
    if len(ids) < 20:
        return None  # TCGA not in training -> no privileged omics available
    return anchoring_score(E.loc[ids, universe], CNA.loc[ids], METH.loc[ids])


def run_split(name, tr, te, seed):
    f = CK / f"{name}.json"
    if f.exists():
        return json.load(open(f))
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ttr, etr, tte, ete, cte = T[tr], EV[tr], T[te], EV[te], COH[te]
    A = Xtr.values
    anc = anchor_from_training(list(X.index[tr]))
    res, ngen = {}, {}

    def score(r):
        return cindex_strat(np.asarray(r), tte, ete, cte)

    zs = np.abs(univariate_cox_z(A, ttr, etr))
    top = np.argsort(-zs)[:10]
    res["Univariate top-10 + ridge Cox"] = score(Xte.values[:, top] @ ridge_cox(A[:, top], ttr, etr, 1.0)); ngen["Univariate top-10 + ridge Cox"] = 10
    for nm, l1 in [("Cox-LASSO", 1.0), ("Cox elastic net", 0.5)]:
        coef = coxnet_cv(A, ttr, etr, l1, seed=seed)
        res[nm] = score(Xte.values @ coef) if np.any(coef) else 0.5; ngen[nm] = int((coef != 0).sum())
    cand = np.argsort(-zs)[:300]
    pr = (zs[cand] - zs[cand].mean()) / zs[cand].std()
    if anc is not None:
        a = anc.reindex(universe[cand]).fillna(0).values
        pr = pr + (a - a.mean()) / (a.std() + 1e-8)
    pf = np.exp(-pr / 2); pf = pf / pf.mean()
    coef = coxnet_cv(A[:, cand], ttr, etr, 1.0, penalty_factor=pf, seed=seed)
    res["Anchor-weighted adaptive LASSO"] = score(Xte.values[:, cand] @ coef) if np.any(coef) else 0.5
    ngen["Anchor-weighted adaptive LASSO"] = int((coef != 0).sum())
    c5 = np.argsort(-zs)[:500]
    rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=5, max_features="sqrt", n_jobs=1,
                               random_state=seed).fit(A[:, c5], Surv.from_arrays(etr.astype(bool), ttr))
    res["Random survival forest"] = score(rsf.predict(Xte.values[:, c5])); ngen["Random survival forest"] = 500

    m = OASIS(seed=seed); m.anchor_override = anc; m.use_anchor = anc is not None
    m.fit(Xtr, ttr, etr)
    res["OASIS"] = score(m.predict_risk(Xte)); ngen["OASIS"] = len(m.selected_)
    m2 = OASIS2(reward="boot", seed=seed); m2.anchor_override = anc; m2.use_anchor = anc is not None
    m2.fit(Xtr, ttr, etr)
    res["OASIS-2 (single signature)"] = score(m2.predict_risk(Xte)); ngen["OASIS-2 (single signature)"] = len(m2.selected_)
    res["OASIS-2 (policy ensemble)"] = score(m2.predict_risk_ensemble(Xte)); ngen["OASIS-2 (policy ensemble)"] = len(m2.ens_genes_)
    m3 = OASIS2(reward="boot", seed=seed, use_anchor=False).fit(Xtr, ttr, etr)
    res["OASIS-2 ensemble w/o anchoring"] = score(m3.predict_risk_ensemble(Xte)); ngen["OASIS-2 ensemble w/o anchoring"] = len(m3.ens_genes_)
    o = dict(split=name, cindex=res, n_genes=ngen, sel_oasis2=m2.selected_, anchored=anc is not None,
             n_train=len(tr), n_test=len(te))
    json.dump(o, open(f, "w"))
    return o


if __name__ == "__main__" and MODE in ("cv", "loco"):
    jobs = []
    if MODE == "cv":
        strat = np.array([f"{c}_{e}" for c, e in zip(COH, EV)])
        for rep in range(R0, R1 + 1):
            for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X.values, strat)):
                jobs.append((f"r{rep}_f{k}", tr, te, 1000 * rep + k))
    else:
        for c in ["TCGA", "GSE22138", "GSE44295"]:
            te = np.where(COH == c)[0]; tr = np.where(COH != c)[0]
            for s in range(5):  # 5 seeds for the stochastic methods
                jobs.append((f"test_{c}_s{s}", tr, te, s))
    out = Parallel(n_jobs=2, verbose=2)(delayed(run_split)(*j) for j in jobs)
    print("done", len(out))
