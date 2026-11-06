"""Additional control analyses, on exactly the same splits as the main analyses.

Methods added
  - Adaptive LASSO, positional prior : the anchoring term is replaced by an indicator of
    genes on chromosome 3, 8q or 1p (the UM risk arms), standardised. Tests whether
    anchoring is more than "pick genes on the risk arms".
  - Adaptive LASSO, no anchoring     : re-run in leave-one-cohort-out with the same five
    seeds as the anchored version (inner-CV seed), so held-out-TCGA results are identical.
  - 15-gene-class proxy              : ridge Cox on the 12 discriminating genes of the
    DecisionDx-UM panel (Onken et al., 2010), trained in the training data;
    LTA4H is absent on GPL6883, so pooled/LOCO/external use the 11 shared genes.
  - 15-gene-class proxy + PRAME      : as above plus PRAME.

Usage: python review_extra.py PROCESSED_DIR GEO_DIR RESULTS_DIR CNA_TXT
"""
import sys, re, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
P, G, R, CNA_TXT = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
sys.argv = [sys.argv[0], P, G, str(R), "info_quiet"]
import pooled as PL
from rl_agent import cindex, univariate_cox_z, ridge_cox
from run_cv_helpers import coxnet_cv

GEP = ["HTR2B", "ID2", "MTUS1", "ECM1", "ROBO1", "SATB1", "LTA4H", "EIF1B", "FXR1", "CDH1", "LMCD1", "RAB31"]
band = {l.split("\t")[0]: l.split("\t")[2].strip() for l in open(CNA_TXT) if not l.startswith("Hugo")}


def on_risk_arm(g):
    m = re.match(r"^(\d+)([pq])", band.get(g, ""))
    return bool(m) and (m.group(1) == "3" or (m.group(1), m.group(2)) in [("8", "q"), ("1", "p")])


def alasso(A, genes, t, e, seed, mode):
    zs = np.abs(univariate_cox_z(A, t, e))
    cand = np.argsort(-zs)[:300]
    pr = (zs[cand] - zs[cand].mean()) / zs[cand].std()
    if mode == "positional":
        a = np.array([on_risk_arm(g) for g in genes[cand]], float)
        if a.std() > 0:
            pr = pr + (a - a.mean()) / a.std()
    pf = np.exp(-pr / 2); pf = pf / pf.mean()
    coef = coxnet_cv(A[:, cand], t, e, 1.0, penalty_factor=pf, seed=seed)
    return cand, coef


def panel(A, genes, t, e, names):
    idx = [list(genes).index(g) for g in names if g in genes]
    return idx, ridge_cox(A[:, idx], t, e, 1.0)


def fit_all(A, Ate, genes, t, e, seed, with_noanchor=True):
    out, sets = {}, {}
    for mode, nm in [("positional", "Adaptive LASSO, positional prior")] + ([("none", "Adaptive LASSO, no anchoring")] if with_noanchor else []):
        cand, coef = alasso(A, genes, t, e, seed, mode)
        out[nm] = Ate[:, cand] @ coef if np.any(coef) else np.zeros(len(Ate))
        sets[nm] = list(genes[cand][coef != 0])
    for nm, names in [("15-gene-class proxy", GEP), ("15-gene-class proxy + PRAME", GEP + ["PRAME"])]:
        idx, b = panel(A, genes, t, e, names)
        out[nm] = Ate[:, idx] @ b
        sets[nm] = [genes[i] for i in idx]
    return out, sets


# ---------- TCGA-only CV (same splits / per-fold standardisation as run_cv.py)
E = pd.read_csv(Path(P) / "expr.csv.gz", index_col=0)
CL = pd.read_csv(Path(P) / "clinical.csv", index_col=0)
T, EV = CL["PFS_months"].values, CL["PFS_event"].values
GENES = np.asarray(E.columns)


def tcga_fold(rep, k, tr, te):
    Xtr, Xte = E.values[tr], E.values[te]
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    risks, sets = fit_all((Xtr - m) / s, (Xte - m) / s, GENES, T[tr], EV[tr], rep, with_noanchor=False)
    return dict(rep=rep, fold=k, cindex={n: cindex(r, T[te], EV[te]) for n, r in risks.items()}, sets=sets)


jobs = [(rep, k, tr, te) for rep in range(10)
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(E.values, EV))]
json.dump(Parallel(n_jobs=2)(delayed(tcga_fold)(*j) for j in jobs), open(R / "review_tcga.json", "w"))
print("tcga done")

# ---------- pooled CV and LOCO (same splits as pooled.py)
UG = np.asarray(PL.universe)


def pooled_split(name, tr, te, seed, with_noanchor):
    risks, sets = fit_all(PL.X.values[tr], PL.X.values[te], UG, PL.T[tr], PL.EV[tr], seed, with_noanchor)
    return dict(split=name, cindex={n: PL.cindex_strat(np.asarray(r), PL.T[te], PL.EV[te], PL.COH[te]) for n, r in risks.items()},
                n_genes={n: len(v) for n, v in sets.items()})


strat = np.array([f"{c}_{e}" for c, e in zip(PL.COH, PL.EV)])
pj = [(f"r{rep}_f{k}", tr, te, 1000 * rep + k, False) for rep in range(10)
      for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(PL.X.values, strat))]
json.dump(Parallel(n_jobs=2)(delayed(pooled_split)(*j) for j in pj), open(R / "review_pooled.json", "w"))
print("pooled done")
lj = [(f"test_{c}_s{s}", np.where(PL.COH != c)[0], np.where(PL.COH == c)[0], s, True)
      for c in ["TCGA", "GSE22138", "GSE44295"] for s in range(5)]
json.dump(Parallel(n_jobs=2)(delayed(pooled_split)(*j) for j in lj), open(R / "review_loco.json", "w"))
print("loco done")
