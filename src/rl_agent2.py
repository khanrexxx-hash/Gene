"""OASIS-2: fixes the weaknesses found in OASIS.

1. Fresh reward splits every iteration (OASIS used 4 fixed bootstraps, which the
   agent could overfit: training reward kept rising while held-out C did not).
2. Optional cross-validated reward: pooled out-of-fold C-index from a freshly
   shuffled inner 3-fold split (lower variance than out-of-bag C on ~30% of patients).
3. Policy ensemble: the final predictor averages ridge-Cox risk scores of M gene
   subsets sampled from the learned policy (bagging over the policy), instead of a
   single refit. A compact signature (p > 0.5) is still reported for interpretation.
"""
import numpy as np
from rl_agent import ridge_cox, cindex, univariate_cox_z, anchoring_score


def _strat_folds(e, k, rng):
    idx = np.arange(len(e))
    f = np.empty(len(e), int)
    for cls in (0, 1):
        c = rng.permutation(idx[e == cls])
        f[c] = np.arange(len(c)) % k
    return f


class OASIS2:
    def __init__(self, reward="cv", n_candidates=300, anchor_weight=1.0, stability_weight=1.0,
                 sparsity=0.004, n_boot=4, n_inner=3, iters=150, batch=12, lr=0.15, entropy=0.01,
                 init_k=8, max_k=20, ridge=1.0, ensemble_M=50, use_anchor=True, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def _sample(self, p, rng):
        m = rng.random(len(p)) < p
        if m.sum() > self.max_k:
            on = np.where(m)[0]
            m[:] = False
            m[on[np.argsort(-p[on])[: self.max_k]]] = True
        return m

    def _reward(self, mask, X, t, e, split):
        k = mask.sum()
        if k == 0:
            return 0.0
        cols = np.where(mask)[0]
        if self.reward == "cv":
            # split = list of fold labels arrays (repeats); pooled out-of-fold C per repeat
            cs = []
            for f in split:
                risk = np.empty(len(t))
                for j in range(self.n_inner):
                    tr, te = f != j, f == j
                    b = ridge_cox(X[np.ix_(tr, cols)], t[tr], e[tr], self.ridge)
                    risk[te] = X[np.ix_(te, cols)] @ b
                    # put folds on a common scale before pooling
                    risk[te] = (risk[te] - risk[te].mean()) / (risk[te].std() + 1e-8)
                cs.append(cindex(risk, t, e))
        else:
            cs = []
            for ib, oob in split:
                b = ridge_cox(X[np.ix_(ib, cols)], t[ib], e[ib], self.ridge)
                cs.append(cindex(X[np.ix_(oob, cols)] @ b, t[oob], e[oob]))
        cs = np.asarray(cs)
        r = cs.mean() - (self.stability_weight * cs.std() if len(cs) > 1 else 0.0)
        return r - self.sparsity * k

    def _new_split(self, rng, n, e):
        if self.reward == "cv":
            return [_strat_folds(e, self.n_inner, rng) for _ in range(2)]
        out = []
        for _ in range(self.n_boot):
            ib = rng.choice(n, n, replace=True)
            out.append((ib, np.setdiff1d(np.arange(n), ib)))
        return out

    def fit(self, expr, time, event, cna=None, meth=None):
        rng = np.random.default_rng(self.seed)
        genes = np.asarray(expr.columns)
        Xall = expr.values.astype(float)
        mu, sd = Xall.mean(0), Xall.std(0) + 1e-8
        Xall = (Xall - mu) / sd
        t, e = np.asarray(time, float), np.asarray(event, int)
        n = len(t)

        z = np.abs(univariate_cox_z(Xall, t, e))
        cand = np.argsort(-z)[: self.n_candidates]
        X = Xall[:, cand]
        zc = z[cand]
        prior = (zc - zc.mean()) / (zc.std() + 1e-8)
        if self.use_anchor and (cna is not None or meth is not None or getattr(self, "anchor_override", None) is not None):
            a = (self.anchor_override.reindex(genes[cand]).fillna(0.0).values
                 if getattr(self, "anchor_override", None) is not None
                 else anchoring_score(expr.iloc[:, cand], cna, meth).values)
            prior = prior + self.anchor_weight * (a - a.mean()) / (a.std() + 1e-8)
        p0 = self.init_k / len(cand)
        theta = prior + np.log(p0 / (1 - p0)) - np.median(prior)

        baseline = None
        self.history_ = []
        for it in range(self.iters):
            split = self._new_split(rng, n, e)  # fresh every iteration -> no split overfitting
            p = 1 / (1 + np.exp(-theta))
            masks = [self._sample(p, rng) for _ in range(self.batch)]
            rewards = np.array([self._reward(m, X, t, e, split) for m in masks])
            baseline = rewards.mean() if baseline is None else 0.9 * baseline + 0.1 * rewards.mean()
            adv = (rewards - baseline) / (rewards.std() + 1e-8)
            g = (adv[:, None] * (np.asarray(masks, float) - p)).mean(0)
            theta += self.lr * (g - self.entropy * p * (1 - p) * theta)
            self.history_.append(rewards.mean())

        p = 1 / (1 + np.exp(-theta))
        self.prob_ = dict(zip(genes[cand], p))
        sel = p > 0.5
        if sel.sum() == 0:
            sel[np.argsort(-p)[: self.init_k]] = True
        if sel.sum() > self.max_k:
            sel[:] = False; sel[np.argsort(-p)[: self.max_k]] = True
        self.sel_idx_ = cand[sel]
        self.selected_ = list(genes[self.sel_idx_])
        self.beta_ = ridge_cox(Xall[:, self.sel_idx_], t, e, self.ridge)

        # policy ensemble
        self.ens_ = []
        for _ in range(self.ensemble_M):
            m = self._sample(p, rng)
            if m.sum() == 0:
                continue
            idx = cand[m]
            b = ridge_cox(Xall[:, idx], t, e, self.ridge)
            r = Xall[:, idx] @ b
            self.ens_.append((idx, b, r.mean(), r.std() + 1e-8))
        self.mu_, self.sd_ = mu, sd
        self.ens_genes_ = sorted({g for idx, *_ in self.ens_ for g in genes[idx]})
        return self

    def _X(self, expr):
        return (expr.values.astype(float) - self.mu_) / self.sd_

    def predict_risk(self, expr):
        return self._X(expr)[:, self.sel_idx_] @ self.beta_

    def predict_risk_ensemble(self, expr):
        X = self._X(expr)
        return np.mean([((X[:, idx] @ b) - m) / s for idx, b, m, s in self.ens_], axis=0)
