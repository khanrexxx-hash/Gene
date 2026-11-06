"""Additional control and sensitivity analyses (same splits as all main analyses).

TCGA 10x5 CV:
  - permuted anchoring (20 permutations of the anchoring score across candidates, per fold)
  - rescaled prior (pi rescaled to unit SD before computing weights)
  - positional prior including 6p
  - sensitivity: mean instead of max of the two correlations; weight scale exp(-pi/1), exp(-pi/4);
    150 and 600 candidates; lambda tuned on held-out partial log-likelihood instead of C-index
  - molecular markers + anchored score (both fitted inside the fold)
Pooled 10x5 CV:
  - permuted anchoring (10 permutations), rescaled prior, positional + 6p
  - cohort-stratified Cox (stratified screen + stratified weighted LASSO) for anchored,
    no-anchoring and positional models; also in leave-one-cohort-out

Usage: python review2.py PROCESSED_DIR GEO_DIR RESULTS_DIR CNA_TXT
"""
import sys, re, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
P, G, R, CNA_TXT = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
sys.argv = [sys.argv[0], P, G, str(R), "info_quiet"]
import pooled as PL
from rl_agent import cindex, univariate_cox_z, ridge_cox
from run_cv_helpers import coxnet_cv
from strat_lasso import strat_lasso_cv, strat_score_z, cindex_strat

band = {l.split("\t")[0]: l.split("\t")[2].strip() for l in open(CNA_TXT) if not l.startswith("Hugo")}


def arm_flag(g, with6p=False):
    m = re.match(r"^(\d+)([pq])", band.get(g, ""))
    if not m:
        return False
    c, a = m.group(1), m.group(2)
    return c == "3" or (c, a) in [("8", "q"), ("1", "p")] or (with6p and (c, a) == ("6", "p"))


def corr_parts(E_c, cna, meth):
    """Pearson |r| of each gene's expression with its own copy number and methylation (NaN if absent)."""
    out = {}
    for name, om in (("cna", cna), ("meth", meth)):
        g = E_c.columns.intersection(om.columns)
        a = E_c[g].values; b = om.loc[E_c.index, g].values.astype(float)
        a = (a - a.mean(0)) / (a.std(0) + 1e-8); b = (b - b.mean(0)) / (b.std(0) + 1e-8)
        out[name] = pd.Series(np.abs((a * b).mean(0)), index=g).reindex(E_c.columns)
    return out["cna"], out["meth"]


def std(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def held_out_pl_cv(Xtr, ttr, etr, pf, seed):
    """Coxnet with lambda chosen by inner 3-fold held-out partial log-likelihood."""
    y = Surv.from_arrays(etr.astype(bool), ttr)
    kw = dict(l1_ratio=1.0, alpha_min_ratio=0.05, n_alphas=30, max_iter=20000, penalty_factor=pf)
    full = CoxnetSurvivalAnalysis(**kw, fit_baseline_model=False).fit(Xtr, y)
    alphas = full.alphas_
    sc = np.zeros(len(alphas))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, etr):
        try:
            m = CoxnetSurvivalAnalysis(**{**kw, "alphas": alphas}).fit(Xtr[a], y[a])
        except Exception:
            continue
        for j in range(len(alphas)):
            coef = m.coef_[:, min(j, m.coef_.shape[1] - 1)]
            eta = Xtr[b] @ coef; tb, eb = ttr[b], etr[b]
            o = np.argsort(-tb); eta, eb = eta[o], eb[o]
            sc[j] += (eb * (eta - np.log(np.cumsum(np.exp(eta))))).sum()
    return full.coef_[:, int(np.argmax(sc))]


def fit_alasso(Xtr, genes, ttr, etr, seed, a_vec=None, scale=2.0, ncand=300, rescale=False,
               tune="c", cand=None, zs=None):
    if zs is None:
        zs = np.abs(univariate_cox_z(Xtr, ttr, etr))
    if cand is None:
        cand = np.argsort(-zs)[:ncand]
    pr = std(zs[cand])
    if a_vec is not None:
        pr = pr + std(a_vec)
    if rescale:
        pr = pr / (pr.std() + 1e-8)
    pf = np.exp(-pr / scale); pf = pf / pf.mean()
    coef = coxnet_cv(Xtr[:, cand], ttr, etr, 1.0, penalty_factor=pf, seed=seed) if tune == "c" else \
        held_out_pl_cv(Xtr[:, cand], ttr, etr, pf, seed)
    return cand, coef


# ======================= TCGA CV
E = pd.read_csv(Path(P) / "expr.csv.gz", index_col=0)
CNA = pd.read_csv(Path(P) / "cna.csv.gz", index_col=0)
METH = pd.read_csv(Path(P) / "meth.csv.gz", index_col=0)
CL = pd.read_csv(Path(P) / "clinical.csv", index_col=0)
T, EV = CL["PFS_months"].values, CL["PFS_event"].values
GENES = np.asarray(E.columns)
MARK = ["chr3_loss", "chr8q_gain", "BAP1_mut"]


def tcga_fold(rep, k, tr, te):
    Xtr, Xte = E.values[tr], E.values[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ttr, etr, tte, ete = T[tr], EV[tr], T[te], EV[te]
    zs = np.abs(univariate_cox_z(Xtr, ttr, etr))
    res = {}

    def ev(name, cand, coef):
        res[name] = cindex(Xte[:, cand] @ coef, tte, ete) if np.any(coef) else 0.5

    def anchor_vec(cand, how="max"):
        rc, rm = corr_parts(E.iloc[tr][GENES[cand]], CNA.iloc[tr], METH.iloc[tr])
        a = (pd.concat([rc, rm], axis=1).max(axis=1) if how == "max" else pd.concat([rc, rm], axis=1).mean(axis=1))
        return a.fillna(0.0).values

    for ncand in (150, 300, 600):
        cand = np.argsort(-zs)[:ncand]
        a = anchor_vec(cand)
        if ncand == 300:
            c0, cf = fit_alasso(Xtr, GENES, ttr, etr, rep, a, cand=cand, zs=zs); ev("anchored", c0, cf)
            anchored_score = (Xtr[:, c0] @ cf, Xte[:, c0] @ cf)
            ev("rescaled prior", *fit_alasso(Xtr, GENES, ttr, etr, rep, a, rescale=True, cand=cand, zs=zs))
            ev("weights exp(-pi/1)", *fit_alasso(Xtr, GENES, ttr, etr, rep, a, scale=1.0, cand=cand, zs=zs))
            ev("weights exp(-pi/4)", *fit_alasso(Xtr, GENES, ttr, etr, rep, a, scale=4.0, cand=cand, zs=zs))
            ev("mean of correlations", *fit_alasso(Xtr, GENES, ttr, etr, rep, anchor_vec(cand, "mean"), cand=cand, zs=zs))
            ev("lambda by partial likelihood", *fit_alasso(Xtr, GENES, ttr, etr, rep, a, tune="pl", cand=cand, zs=zs))
            pos6 = np.array([arm_flag(g, True) for g in GENES[cand]], float)
            ev("positional prior + 6p", *fit_alasso(Xtr, GENES, ttr, etr, rep, pos6, cand=cand, zs=zs))
            rng = np.random.default_rng(1000 * rep + k)
            perm = [cindex(Xte[:, c] @ f, tte, ete) if np.any(f) else 0.5
                    for c, f in [fit_alasso(Xtr, GENES, ttr, etr, rep, rng.permutation(a), cand=cand, zs=zs) for _ in range(20)]]
            res["permuted anchoring (mean of 20)"] = float(np.mean(perm))
        else:
            ev(f"{ncand} candidates", *fit_alasso(Xtr, GENES, ttr, etr, rep, a, cand=cand, zs=zs))

    # markers alone and markers + anchored score, all fitted on the training fold
    Mtr, Mte = CL.iloc[tr][MARK].values.astype(float), CL.iloc[te][MARK].values.astype(float)
    m_mu, m_sd = Mtr.mean(0), Mtr.std(0) + 1e-8
    Mtr, Mte = (Mtr - m_mu) / m_sd, (Mte - m_mu) / m_sd
    s_tr, s_te = anchored_score
    s_mu, s_sd = s_tr.mean(), s_tr.std() + 1e-8
    Ztr = np.c_[Mtr, (s_tr - s_mu) / s_sd]; Zte = np.c_[Mte, (s_te - s_mu) / s_sd]
    res["markers + anchored score"] = cindex(Zte @ ridge_cox(Ztr, ttr, etr, 1.0), tte, ete)
    res["markers alone (refit)"] = cindex(Mte @ ridge_cox(Mtr, ttr, etr, 1.0), tte, ete)
    return dict(rep=rep, fold=k, cindex=res)


jobs = [(rep, k, tr, te) for rep in range(10)
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(E.values, EV))]
json.dump(Parallel(n_jobs=2)(delayed(tcga_fold)(*j) for j in jobs), open(R / "review2_tcga.json", "w"))
print("tcga done", flush=True)

# ======================= pooled CV / LOCO
UG = np.asarray(PL.universe)
COH = PL.COH


def pooled_split(name, tr, te, seed, do_perm=True):
    A, Ate = PL.X.values[tr], PL.X.values[te]
    ttr, etr, ctr = PL.T[tr], PL.EV[tr], COH[tr]
    tte, ete, cte = PL.T[te], PL.EV[te], COH[te]
    anc = PL.anchor_from_training(list(PL.X.index[tr]))
    res = {}
    # --- unstratified (as in the main analysis)
    zs = np.abs(univariate_cox_z(A, ttr, etr)); cand = np.argsort(-zs)[:300]
    a = anc.reindex(UG[cand]).fillna(0).values if anc is not None else None
    pos = np.array([arm_flag(g) for g in UG[cand]], float)
    pos6 = np.array([arm_flag(g, True) for g in UG[cand]], float)

    def ev(nm, c, f):
        res[nm] = cindex_strat(Ate[:, c] @ f, tte, ete, cte) if np.any(f) else 0.5
    if a is not None:
        ev("rescaled prior", *fit_alasso(A, UG, ttr, etr, seed, a, rescale=True, cand=cand, zs=zs))
        if do_perm:
            rng = np.random.default_rng(seed)
            res["permuted anchoring (mean of 10)"] = float(np.mean([
                (lambda cf: cindex_strat(Ate[:, cf[0]] @ cf[1], tte, ete, cte) if np.any(cf[1]) else 0.5)(
                    fit_alasso(A, UG, ttr, etr, seed, rng.permutation(a), cand=cand, zs=zs)) for _ in range(10)]))
    ev("positional prior + 6p", *fit_alasso(A, UG, ttr, etr, seed, pos6, cand=cand, zs=zs))
    # --- cohort-stratified screen + stratified weighted LASSO
    zs_s = np.abs(strat_score_z(A, ttr, etr, ctr)); cand_s = np.argsort(-zs_s)[:300]
    pr0 = std(zs_s[cand_s])
    variants = {"stratified: no anchoring": pr0,
                "stratified: positional prior": pr0 + std(np.array([arm_flag(g) for g in UG[cand_s]], float))}
    if anc is not None:
        variants["stratified: anchored"] = pr0 + std(anc.reindex(UG[cand_s]).fillna(0).values)
    else:
        variants["stratified: anchored"] = pr0  # no anchoring data when TCGA is held out
    for nm, pr in variants.items():
        pf = np.exp(-pr / 2); pf = pf / pf.mean()
        cf = strat_lasso_cv(A[:, cand_s], ttr, etr, ctr, pf, seed=seed)
        res[nm] = cindex_strat(Ate[:, cand_s] @ cf, tte, ete, cte) if np.any(cf) else 0.5
    return dict(split=name, cindex=res)


strat = np.array([f"{c}_{e}" for c, e in zip(COH, PL.EV)])
pj = [(f"r{rep}_f{k}", tr, te, 1000 * rep + k) for rep in range(10)
      for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(PL.X.values, strat))]
json.dump(Parallel(n_jobs=2)(delayed(pooled_split)(*j) for j in pj), open(R / "review2_pooled.json", "w"))
print("pooled done", flush=True)
lj = [(f"test_{c}_s{s}", np.where(COH != c)[0], np.where(COH == c)[0], s, False)
      for c in ["TCGA", "GSE22138", "GSE44295"] for s in range(5)]
json.dump(Parallel(n_jobs=2)(delayed(pooled_split)(*j) for j in lj), open(R / "review2_loco.json", "w"))
print("loco done", flush=True)
