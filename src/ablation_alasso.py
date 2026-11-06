"""Control for the anchoring prior: adaptive LASSO with univariate-only penalty weights.

Identical to 'Anchor-weighted adaptive LASSO' except the anchoring term is removed, on the
same splits as every other analysis. Writes results/alasso_noanchor_{tcga,pooled,loco}.csv

Usage: python ablation_alasso.py PROCESSED_DIR GEO_DIR RESULTS_DIR
"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
P, G, R = map(Path, sys.argv[1:4])
sys.argv = [sys.argv[0], str(P), str(G), str(R), "info_quiet"]
import pooled as PL  # reuses the pooled cohort assembly (no jobs run in this mode)
from rl_agent import cindex, univariate_cox_z
from run_cv_helpers import coxnet_cv


def alasso(A, t, e, seed):
    zs = np.abs(univariate_cox_z(A, t, e))
    cand = np.argsort(-zs)[:300]
    pr = (zs[cand] - zs[cand].mean()) / zs[cand].std()
    pf = np.exp(-pr / 2); pf = pf / pf.mean()
    return cand, coxnet_cv(A[:, cand], t, e, 1.0, penalty_factor=pf, seed=seed)


# --- TCGA-only CV (same splits as run_cv.py; per-fold standardisation as there)
E = pd.read_csv(P / "expr.csv.gz", index_col=0)
CL = pd.read_csv(P / "clinical.csv", index_col=0)
T, EV = CL["PFS_months"].values, CL["PFS_event"].values


def tcga_fold(rep, k, tr, te):
    Xtr, Xte = E.values[tr], E.values[te]
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - m) / s, (Xte - m) / s
    cand, coef = alasso(Xtr, T[tr], EV[tr], rep)
    c = cindex(Xte[:, cand] @ coef, T[te], EV[te]) if np.any(coef) else 0.5
    return dict(rep=rep, fold=k, cindex=c, n_genes=int((coef != 0).sum()), genes=list(E.columns[cand][coef != 0]))


jobs = [(rep, k, tr, te) for rep in range(10)
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(E.values, EV))]
out = Parallel(n_jobs=2)(delayed(tcga_fold)(*j) for j in jobs)
pd.DataFrame(out).to_json(R / "alasso_noanchor_tcga.json", orient="records")


# --- pooled CV and LOCO (same splits as pooled.py)
def pooled_split(name, tr, te, seed):
    A = PL.X.values[tr]
    cand, coef = alasso(A, PL.T[tr], PL.EV[tr], seed)
    r = PL.X.values[te][:, cand] @ coef
    c = PL.cindex_strat(r, PL.T[te], PL.EV[te], PL.COH[te]) if np.any(coef) else 0.5
    return dict(split=name, cindex=c, n_genes=int((coef != 0).sum()))


strat = np.array([f"{c}_{e}" for c, e in zip(PL.COH, PL.EV)])
pj = [(f"r{rep}_f{k}", tr, te, 1000 * rep + k) for rep in range(10)
      for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(PL.X.values, strat))]
pd.DataFrame(Parallel(n_jobs=2)(delayed(pooled_split)(*j) for j in pj)).to_csv(R / "alasso_noanchor_pooled.csv", index=False)
lj = [(f"test_{c}_s0", np.where(PL.COH != c)[0], np.where(PL.COH == c)[0], 0) for c in ["TCGA", "GSE22138", "GSE44295"]]
pd.DataFrame([pooled_split(*j) for j in lj]).to_csv(R / "alasso_noanchor_loco.csv", index=False)
print("done")
