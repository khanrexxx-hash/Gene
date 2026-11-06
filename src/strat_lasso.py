"""Cohort-stratified weighted LASSO Cox model (proximal gradient / FISTA), with a stratified
univariate screen and stratified C-index. Used for the pooled designs so that each cohort keeps
its own baseline hazard.
"""
import numpy as np


def _prep(t, e, strata):
    groups = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        o = idx[np.argsort(-t[idx], kind="mergesort")]  # descending time: risk sets are prefixes
        groups.append((o, e[o].astype(float)))
    return groups


def neg_loglik_grad(beta, X, groups):
    """Negative stratified Breslow partial log-likelihood (per observation) and gradient."""
    n = X.shape[0]
    ll, g = 0.0, np.zeros(X.shape[1])
    for o, d in groups:
        Xo = X[o]
        eta = np.clip(Xo @ beta, -30, 30)
        w = np.exp(eta)
        cw = np.cumsum(w)
        cwx = np.cumsum(w[:, None] * Xo, axis=0)
        ll += (d * (eta - np.log(cw))).sum()
        g += (d[:, None] * (Xo - cwx / cw[:, None])).sum(0)
    return -ll / n, -g / n


def lasso_path(X, t, e, strata, pf, n_lam=20, lam_min_ratio=0.05, iters=300, tol=1e-6, lams=None):
    """Weighted-L1 Cox path. pf: penalty factors (mean 1). Returns lambdas, coef matrix (p x n_lam)."""
    groups = _prep(t, e, strata)
    p = X.shape[1]
    _, g0 = neg_loglik_grad(np.zeros(p), X, groups)
    if lams is None:
        lam_max = np.max(np.abs(g0) / np.maximum(pf, 1e-8))
        lams = lam_max * np.logspace(0, np.log10(lam_min_ratio), n_lam)
    L = 0.25 * np.max(np.sum(X ** 2, 1))  # crude Lipschitz bound
    beta = np.zeros(p)
    coefs = np.zeros((p, n_lam))
    for k, lam in enumerate(lams):
        z, b_old, tk = beta.copy(), beta.copy(), 1.0
        for _ in range(iters):
            _, gz = neg_loglik_grad(z, X, groups)
            u = z - gz / L
            b_new = np.sign(u) * np.maximum(np.abs(u) - lam * pf / L, 0)
            t_new = (1 + np.sqrt(1 + 4 * tk ** 2)) / 2
            z = b_new + (tk - 1) / t_new * (b_new - b_old)
            if np.max(np.abs(b_new - b_old)) < tol:
                b_old = b_new
                break
            b_old, tk = b_new, t_new
        beta = b_old
        coefs[:, k] = beta
    return lams, coefs


def cindex_strat(r, t, e, strata):
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        rr, tt, ee = r[m], t[m], e[m]
        for i in np.where(ee == 1)[0]:
            comp = tt > tt[i]
            num += (rr[i] > rr[comp]).sum() + 0.5 * (rr[i] == rr[comp]).sum()
            den += comp.sum()
    return num / den if den else 0.5


def strat_score_z(X, t, e, strata):
    """Stratified univariate Cox score statistic (sum of per-stratum U and V)."""
    U = np.zeros(X.shape[1]); V = np.zeros(X.shape[1])
    for o, d in _prep(t, e, strata):
        Xo = X[o]
        cnt = np.arange(1, len(o) + 1)
        mu = np.cumsum(Xo, 0) / cnt[:, None]
        mu2 = np.cumsum(Xo ** 2, 0) / cnt[:, None]
        dd = d.astype(bool)
        U += (Xo[dd] - mu[dd]).sum(0)
        V += (mu2[dd] - mu[dd] ** 2).sum(0)
    return U / np.sqrt(V + 1e-12)


def strat_lasso_cv(X, t, e, strata, pf, seed=0, k=3):
    """Choose lambda by inner k-fold CV (stratified C-index), refit on all; return coef."""
    rng = np.random.default_rng(seed)
    lams, full = lasso_path(X, t, e, strata, pf)
    fold = np.empty(len(t), int)
    for s in np.unique(strata):
        for ev in (0, 1):
            idx = rng.permutation(np.where((strata == s) & (e == ev))[0])
            fold[idx] = np.arange(len(idx)) % k
    score = np.zeros(len(lams))
    for j in range(k):
        tr, te = fold != j, fold == j
        _, cf = lasso_path(X[tr], t[tr], e[tr], strata[tr], pf, lams=lams)
        for li in range(len(lams)):
            b = cf[:, li]
            score[li] += cindex_strat(X[te] @ b, t[te], e[te], strata[te]) if np.any(b) else 0.5
    return full[:, int(np.argmax(score))]
