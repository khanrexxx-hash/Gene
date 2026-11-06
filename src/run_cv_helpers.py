"""Shared helpers (importable without running the CV experiment)."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from rl_agent import cindex


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
            scores[j] += cindex(Xtr[b] @ coef, ttr[b], etr[b]) if np.any(coef) else 0.5
    return full.coef_[:, int(np.argmax(scores))]
