"""OASIS: Omics-Anchored, Stability-rewarded Interpretable Selection.

A reinforcement-learning agent that selects a compact expression signature for
survival prediction in very small cohorts.

Key ideas
1. Multi-omics as *privileged information*: copy number and methylation are used
   only at training time to build an "anchoring" prior (genes whose expression is
   explained by their own cis copy-number / methylation state). The final model
   needs expression only, so it can be applied to external microarray cohorts.
2. A stochastic Bernoulli policy over candidate genes, initialised from the prior
   and optimised with REINFORCE (moving-average baseline, entropy bonus).
3. A stability-shaped reward: mean out-of-bag C-index across bootstrap resamples,
   minus its standard deviation (penalises lucky subsets), minus a sparsity cost.
All computation is CPU-only numpy.
"""
import numpy as np


# ---------------------------------------------------------------- Cox utilities
def _order(time):
    return np.argsort(-time, kind="mergesort")  # descending time -> risk sets are prefixes


def ridge_cox(X, time, event, lam=1.0, iters=25):
    """Ridge-penalised Cox (Breslow) by Newton-Raphson. X standardised."""
    o = _order(time)
    X, event = X[o], event[o]
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        w = np.exp(eta)
        cw = np.cumsum(w)
        cwx = np.cumsum(w[:, None] * X, axis=0)
        mu = cwx / cw[:, None]
        grad = (event[:, None] * (X - mu)).sum(0) - lam * beta
        d = event.astype(bool)
        # Hessian: sum over events of weighted covariance in risk set
        H = np.zeros((p, p))
        cwxx = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], axis=0)
        for i in np.where(d)[0]:
            H += cwxx[i] / cw[i] - np.outer(mu[i], mu[i])
        H += lam * np.eye(p)
        step = np.linalg.solve(H, grad)
        beta += step
        if np.abs(step).max() < 1e-6:
            break
    return beta


def cindex(risk, time, event):
    """Harrell's C-index (ties in risk count 0.5)."""
    num = den = 0.0
    idx = np.where(event == 1)[0]
    for i in idx:
        comp = time > time[i]
        if not comp.any():
            continue
        r = risk[comp]
        num += (risk[i] > r).sum() + 0.5 * (risk[i] == r).sum()
        den += comp.sum()
    return num / den if den > 0 else 0.5


def univariate_cox_z(X, time, event):
    """Score-test z statistic per column (fast univariate Cox screen)."""
    o = _order(time)
    X, event = X[o], event[o]
    n = len(time)
    cnt = np.arange(1, n + 1)
    mu = np.cumsum(X, 0) / cnt[:, None]
    mu2 = np.cumsum(X ** 2, 0) / cnt[:, None]
    d = event.astype(bool)
    U = (X[d] - mu[d]).sum(0)
    V = (mu2[d] - mu[d] ** 2).sum(0)
    return U / np.sqrt(V + 1e-12)


def standardise(train, *others):
    m, s = train.mean(0), train.std(0) + 1e-8
    return [(train - m) / s] + [(o - m) / s for o in others]


# ---------------------------------------------------------------- anchoring prior
def anchoring_score(expr, cna=None, meth=None):
    """|corr(expression, own copy number)| and |corr(expression, own methylation)|.

    expr/cna/meth: DataFrames (samples x genes) restricted to the TRAINING fold.
    Returns a Series over expr columns, in [0, 1].
    """
    import pandas as pd
    out = pd.Series(0.0, index=expr.columns)
    parts = []
    for omic in (cna, meth):
        if omic is None:
            continue
        g = expr.columns.intersection(omic.columns)
        a = expr[g].values
        b = omic[g].values.astype(float)
        a = (a - a.mean(0)) / (a.std(0) + 1e-8)
        b = (b - b.mean(0)) / (b.std(0) + 1e-8)
        r = np.abs((a * b).mean(0))
        parts.append(pd.Series(r, index=g).reindex(expr.columns).fillna(0.0))
    if parts:
        out = pd.concat(parts, axis=1).max(axis=1)
    return out


# ---------------------------------------------------------------- the agent
class OASIS:
    def __init__(self, n_candidates=300, anchor_weight=1.0, stability_weight=1.0,
                 sparsity=0.004, n_boot=4, iters=150, batch=12, lr=0.15,
                 entropy=0.01, init_k=8, max_k=20, ridge=1.0, use_anchor=True,
                 use_stability=True, random_prior=False, shift_sigma=0.0, n_shift=2, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # -- reward ----------------------------------------------------------------
    def _reward(self, mask, X, t, e, boots):
        k = mask.sum()
        if k == 0:
            return 0.0
        cols = np.where(mask)[0]
        cs = []
        for j, (ib, oob) in enumerate(boots):
            b = ridge_cox(X[np.ix_(ib, cols)], t[ib], e[ib], self.ridge)
            Xo = X[np.ix_(oob, cols)]
            cs.append(cindex(Xo @ b, t[oob], e[oob]))
            # platform-shift robustness: score on distorted copies of the OOB data
            for sc, sh, nz in self._shifts[j]:
                Xp = Xo * sc[cols] + sh[cols] + nz[: len(oob), cols]
                Xp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)  # re-standardised, as on a new platform
                cs.append(cindex(Xp @ b, t[oob], e[oob]))
        cs = np.asarray(cs)
        r = cs.mean()
        if self.use_stability:
            r -= self.stability_weight * cs.std()
        return r - self.sparsity * k

    # -- fit ---------------------------------------------------------------------
    def fit(self, expr, time, event, cna=None, meth=None):
        rng = np.random.default_rng(self.seed)
        genes = np.asarray(expr.columns)
        Xall = expr.values.astype(float)
        Xall = (Xall - Xall.mean(0)) / (Xall.std(0) + 1e-8)
        t, e = np.asarray(time, float), np.asarray(event, int)

        # 1) candidate screen (training fold only)
        z = np.abs(univariate_cox_z(Xall, t, e))
        cand = np.argsort(-z)[: self.n_candidates]
        X = Xall[:, cand]
        zc = z[cand]

        # 2) prior logits
        prior = (zc - zc.mean()) / (zc.std() + 1e-8)
        if self.use_anchor and (cna is not None or meth is not None or getattr(self, "anchor_override", None) is not None):
            a = (self.anchor_override.reindex(genes[cand]).fillna(0.0).values
                 if getattr(self, "anchor_override", None) is not None
                 else anchoring_score(expr.iloc[:, cand], cna, meth).values)
            prior = prior + self.anchor_weight * (a - a.mean()) / (a.std() + 1e-8)
        if self.random_prior:
            prior = rng.standard_normal(len(cand))
        # calibrate so the expected subset size is ~init_k
        p0 = self.init_k / len(cand)
        offset = np.log(p0 / (1 - p0)) - np.median(prior)
        theta = prior + offset

        # 3) bootstrap splits for the reward (fixed per fit -> lower variance)
        n = len(t)
        boots = []
        for _ in range(self.n_boot):
            ib = rng.choice(n, n, replace=True)
            oob = np.setdiff1d(np.arange(n), ib)
            boots.append((ib, oob))
        # fixed random platform distortions per bootstrap (gene-wise log-scale, shift, noise)
        C = len(cand)
        self._shifts = []
        for _ in boots:
            S = []
            if self.shift_sigma > 0:
                for _ in range(self.n_shift):
                    sc = np.exp(rng.normal(0, self.shift_sigma, C))
                    sh = rng.normal(0, self.shift_sigma, C)
                    nz = rng.normal(0, self.shift_sigma, (n, C))
                    S.append((sc, sh, nz))
            self._shifts.append(S)

        # 4) REINFORCE
        baseline, best = None, (-np.inf, None)
        self.history_ = []
        for it in range(self.iters):
            p = 1 / (1 + np.exp(-theta))
            grads, rewards = [], []
            for _ in range(self.batch):
                m = rng.random(len(p)) < p
                if m.sum() > self.max_k:  # keep the most probable max_k
                    on = np.where(m)[0]
                    m[:] = False
                    m[on[np.argsort(-p[on])[: self.max_k]]] = True
                r = self._reward(m, X, t, e, boots)
                rewards.append(r)
                grads.append(m.astype(float) - p)
                if r > best[0]:
                    best = (r, m.copy())
            rewards = np.asarray(rewards)
            baseline = rewards.mean() if baseline is None else 0.9 * baseline + 0.1 * rewards.mean()
            adv = (rewards - baseline) / (rewards.std() + 1e-8)
            g = (adv[:, None] * np.asarray(grads)).mean(0)
            ent_grad = -p * (1 - p) * theta  # gradient of Bernoulli entropy wrt logits
            theta += self.lr * (g + self.entropy * ent_grad)
            self.history_.append(rewards.mean())

        p = 1 / (1 + np.exp(-theta))
        sel = p > 0.5
        if sel.sum() == 0:
            sel = best[1]
        if sel.sum() > self.max_k:
            keep = np.argsort(-p)[: self.max_k]
            sel = np.zeros_like(sel); sel[keep] = True
        self.selected_ = list(genes[cand][sel])
        self.prob_ = dict(zip(genes[cand], p))
        Xs = expr[self.selected_].values.astype(float)
        self.mu_, self.sd_ = Xs.mean(0), Xs.std(0) + 1e-8
        self.beta_ = ridge_cox((Xs - self.mu_) / self.sd_, t, e, self.ridge)
        return self

    def predict_risk(self, expr):
        Xs = expr[self.selected_].values.astype(float)
        return ((Xs - self.mu_) / self.sd_) @ self.beta_


class OASISPlus:
    """Multi-agent OASIS: n_agents independent agents (own bootstraps / seeds) with a
    platform-shift-robust reward; inclusion probabilities are averaged and thresholded
    at 0.5 to give one consensus signature, refitted by ridge-Cox."""

    def __init__(self, n_agents=5, shift_sigma=0.3, seed=0, **kw):
        self.n_agents, self.shift_sigma, self.seed, self.kw = n_agents, shift_sigma, seed, kw

    def fit(self, expr, time, event, cna=None, meth=None):
        t, e = np.asarray(time, float), np.asarray(event, int)
        probs = {}
        self.agents_ = []
        for a in range(self.n_agents):
            m = OASIS(shift_sigma=self.shift_sigma, seed=self.seed * 97 + a, **self.kw)
            m.fit(expr, t, e, cna, meth)
            self.agents_.append(m)
            for g, p in m.prob_.items():
                probs.setdefault(g, []).append(p)
        pbar = {g: np.sum(v) / self.n_agents for g, v in probs.items()}  # absent from a pool -> p = 0
        ranked = sorted(pbar, key=pbar.get, reverse=True)
        sel = [g for g in ranked if pbar[g] > 0.5][: self.kw.get("max_k", 20)]
        if not sel:
            sel = ranked[: self.kw.get("init_k", 8)]
        self.selected_, self.prob_ = sel, pbar
        self.history_ = list(np.mean([m.history_ for m in self.agents_], 0))
        Xs = expr[sel].values.astype(float)
        self.mu_, self.sd_ = Xs.mean(0), Xs.std(0) + 1e-8
        self.beta_ = ridge_cox((Xs - self.mu_) / self.sd_, t, e, self.kw.get("ridge", 1.0))
        return self

    def predict_risk(self, expr):
        Xs = expr[self.selected_].values.astype(float)
        return ((Xs - self.mu_) / self.sd_) @ self.beta_
