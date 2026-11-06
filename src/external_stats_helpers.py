"""Small helpers: unpenalised Cox fit (Breslow ties, Newton-Raphson) and Kaplan-Meier at a horizon."""
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator


def cox_fit(Xm, t, e, iters=50):
    o = np.argsort(-t, kind="mergesort")
    Xm, e = np.asarray(Xm, float)[o], np.asarray(e)[o]
    beta = np.zeros(Xm.shape[1])
    for _ in range(iters):
        w = np.exp(Xm @ beta)
        cw = np.cumsum(w)
        mu = np.cumsum(w[:, None] * Xm, 0) / cw[:, None]
        cxx = np.cumsum(w[:, None, None] * Xm[:, :, None] * Xm[:, None, :], 0)
        gr = (e[:, None] * (Xm - mu)).sum(0)
        Hm = sum(cxx[i] / cw[i] - np.outer(mu[i], mu[i]) for i in np.where(e == 1)[0])
        step = np.linalg.solve(Hm, gr)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    return beta, np.sqrt(np.diag(np.linalg.inv(Hm)))


def km_at(t, e, h):
    x, s = kaplan_meier_estimator(np.asarray(e).astype(bool), np.asarray(t, float))
    k = np.searchsorted(x, h, side="right") - 1
    return 1.0 if k < 0 else float(s[k])
