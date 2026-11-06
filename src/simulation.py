"""Semi-synthetic simulation on real TCGA-UVM expression, copy number and methylation.

Outcomes are simulated from 5 or 20 causal genes chosen under four scenarios, keeping the real
correlation structure of the omics data:
  S1 known risk arms     : causal genes strongly anchored (|r| > 0.5) and on chr3 / 8q / 1p
  S2 unknown regions     : causal genes strongly anchored but NOT on chr3 / 8q / 1p / 6p
  S3 non-genomic drivers : causal genes weakly anchored (|r| < 0.15), any location
  S4 mixed               : 40% strongly anchored genes off the risk arms + 60% weakly anchored genes
Linear predictor SD = 1 (true C about 0.75); exponential times; uniform censoring scaled to give
37% events, as in TCGA-UVM. Each replicate: one stratified 5-fold CV of four selectors:
no anchoring, anchored, positional (chr3/8q/1p) and permuted anchoring.
Outputs: test C-index and recall / precision of the causal genes.

Usage: python simulation.py PROCESSED_DIR RESULTS_DIR CNA_TXT N_REPS
"""
import sys, re, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
from rl_agent import cindex, univariate_cox_z
from run_cv_helpers import coxnet_cv

P, R, CNA_TXT, NREP = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
E = pd.read_csv(P / "expr.csv.gz", index_col=0)
CNA = pd.read_csv(P / "cna.csv.gz", index_col=0)
METH = pd.read_csv(P / "meth.csv.gz", index_col=0)
GENES = np.asarray(E.columns)
band = {l.split("\t")[0]: l.split("\t")[2].strip() for l in open(CNA_TXT) if not l.startswith("Hugo")}


def arm(g):
    m = re.match(r"^(\d+)([pq])", band.get(g, ""))
    return (m.group(1), m.group(2)) if m else None


RISK = {("3", "p"), ("3", "q"), ("8", "q"), ("1", "p")}
on_risk = np.array([arm(g) in RISK for g in GENES])
off_all = np.array([arm(g) is not None and arm(g) not in RISK | {("6", "p")} for g in GENES])


def corr_abs(Ex, om):
    g = Ex.columns.intersection(om.columns)
    a = Ex[g].values; b = om.loc[Ex.index, g].values.astype(float)
    a = (a - a.mean(0)) / (a.std(0) + 1e-8); b = (b - b.mean(0)) / (b.std(0) + 1e-8)
    return pd.Series(np.abs((a * b).mean(0)), index=g).reindex(Ex.columns)


def anchor_all(idx):
    Ex = E.iloc[idx]
    return pd.concat([corr_abs(Ex, CNA.iloc[idx]), corr_abs(Ex, METH.iloc[idx])], axis=1).max(axis=1).fillna(0.0)


A_FULL = anchor_all(np.arange(len(E))).values  # used only to define scenarios
X = E.values.astype(float)
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
expressed = X.std(0) > 0
POOLS = {
    "S1 known risk arms": np.where((A_FULL > 0.5) & on_risk)[0],
    "S2 unknown regions": np.where((A_FULL > 0.5) & off_all)[0],
    "S3 non-genomic drivers": np.where(A_FULL < 0.15)[0],
}
POOLS["S4 mixed"] = (POOLS["S2 unknown regions"], POOLS["S3 non-genomic drivers"])


def simulate(scen, ncaus, rng):
    if scen == "S4 mixed":
        a_pool, b_pool = POOLS[scen]
        na = round(0.4 * ncaus)
        causal = np.r_[rng.choice(a_pool, na, replace=False), rng.choice(b_pool, ncaus - na, replace=False)]
    else:
        causal = rng.choice(POOLS[scen], ncaus, replace=False)
    beta = rng.choice([-1, 1], ncaus) * rng.uniform(0.5, 1.0, ncaus)
    lp = X[:, causal] @ beta
    lp = (lp - lp.mean()) / lp.std()
    t_ev = rng.exponential(np.exp(-lp)) * 30
    u = rng.uniform(0, 1, len(lp))
    lo, hi = 1e-3, 1e4  # administrative-style censoring scale chosen to give about 37% events (TCGA: 29/79)
    for _ in range(60):
        c = np.sqrt(lo * hi)
        if np.mean(t_ev <= u * c) > 0.37: hi = c
        else: lo = c
    cens = u * c
    t = np.minimum(t_ev, cens); e = (t_ev <= cens).astype(int)
    return causal, lp, np.maximum(t, 0.1), e


def fit(Xtr, ttr, etr, prior_extra, seed):
    zs = np.abs(univariate_cox_z(Xtr, ttr, etr)); cand = np.argsort(-zs)[:300]
    pr = (zs[cand] - zs[cand].mean()) / zs[cand].std()
    if prior_extra is not None:
        v = prior_extra(cand)
        if v.std() > 0:
            pr = pr + (v - v.mean()) / v.std()
    pf = np.exp(-pr / 2); pf = pf / pf.mean()
    return cand, coxnet_cv(Xtr[:, cand], ttr, etr, 1.0, penalty_factor=pf, seed=seed)


CORR = None


def near(sel, causal, thr=0.7):
    """causal genes recovered either exactly or through a selected gene with |r| >= thr (same co-varying block)"""
    if not sel: return 0.0
    s = np.array(sorted(sel))
    r = np.abs(X[:, causal].T @ X[:, s]) / len(X)
    return float(np.mean(r.max(1) >= thr))


def replicate(scen, ncaus, r):
    rng = np.random.default_rng(10_000 * (list(POOLS).index(scen) + 1) + 1000 * ncaus + r)
    causal, lp, t, e = simulate(scen, ncaus, rng)
    out = {m: {"c": [], "rec": [], "prec": [], "rec_block": [], "size": []} for m in ["no anchoring", "anchored", "positional", "permuted anchoring"]}
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(X, e)):
        a_tr = anchor_all(tr).values
        priors = {"no anchoring": None,
                  "anchored": lambda c: a_tr[c],
                  "positional": lambda c: on_risk[c].astype(float),
                  "permuted anchoring": lambda c: rng.permutation(a_tr[c])}
        for m, pe in priors.items():
            cand, cf = fit(X[tr], t[tr], e[tr], pe, r * 10 + k)
            sel = set(cand[cf != 0])
            out[m]["c"].append(cindex(X[te][:, cand] @ cf, t[te], e[te]) if np.any(cf) else 0.5)
            out[m]["rec"].append(len(sel & set(causal)) / ncaus)
            out[m]["rec_block"].append(near(sel, causal))
            out[m]["size"].append(len(sel))
            out[m]["prec"].append(len(sel & set(causal)) / len(sel) if sel else 0.0)
    return dict(scenario=scen, n_causal=ncaus, rep=r, events=int(e.sum()), true_c=cindex(lp, t, e),
                res={m: {k2: float(np.mean(v)) for k2, v in d.items()} for m, d in out.items()})


if __name__ == "__main__":
    print({k: (len(v) if not isinstance(v, tuple) else (len(v[0]), len(v[1]))) for k, v in POOLS.items()}, flush=True)
    jobs = [(s, n, r) for s in POOLS for n in (5, 20) for r in range(NREP)]
    out = Parallel(n_jobs=int(__import__("os").environ.get("NJOBS", "2")), verbose=2)(delayed(replicate)(*j) for j in jobs)
    json.dump(out, open(R / "simulation.json", "w"))
    rows = [dict(scenario=o["scenario"], n_causal=o["n_causal"], rep=o["rep"], method=m, **v) for o in out for m, v in o["res"].items()]
    d = pd.DataFrame(rows)
    summ = d.groupby(["scenario", "n_causal", "method"]).agg(C=("c", "mean"), C_se=("c", lambda x: x.std() / np.sqrt(len(x))),
                                                 recall=("rec", "mean"), recall_block=("rec_block", "mean"),
                                                 precision=("prec", "mean"), size=("size", "mean")).reset_index()
    # paired difference anchored - no anchoring (and permuted, positional) per replicate
    w = d.pivot_table(index=["scenario", "n_causal", "rep"], columns="method", values="c")
    diffs = []
    for (sc, n), g in w.groupby(level=[0, 1]):
        for m in ["no anchoring", "permuted anchoring", "positional"]:
            dd = g["anchored"] - g[m]
            diffs.append(dict(scenario=sc, n_causal=n, vs=m, mean_diff=dd.mean(), se=dd.std() / np.sqrt(len(dd)),
                              wins=(dd > 0).mean()))
    pd.DataFrame(diffs).to_csv(R / "simulation_paired.csv", index=False)
    print(pd.DataFrame(diffs).round(3).to_string(index=False))
    summ.to_csv(R / "simulation_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    ev = pd.DataFrame([dict(s=o["scenario"], n=o["n_causal"], ev=o["events"], tc=o["true_c"]) for o in out]).groupby(["s", "n"]).mean()
    print(ev.round(2))
