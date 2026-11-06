"""Fixed-model external evaluation: train once on all TCGA-UVM, apply unchanged to GSE22138 and GSE44295.

Usage: python external.py PROCESSED_DIR GEO_DIR OUT_DIR
GEO_DIR must contain GSE22138.txt.gz, GSE44295.txt.gz, GPL570.annot.gz, GPL6883.annot.gz

Cross-platform handling: probes collapsed to genes (highest-mean probe); the gene
universe is restricted to genes measured on all three platforms BEFORE training;
each cohort is z-scored per gene within cohort (no information crosses cohorts).
"""
import sys, gzip, io, json, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rl_agent import OASIS, ridge_cox, cindex, univariate_cox_z, anchoring_score
from run_cv_helpers import coxnet_cv

warnings.filterwarnings("ignore")
P, G, OUT = map(Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)


def read_annot(path):
    txt = gzip.open(path, "rt", errors="ignore").read()
    start = txt.index("!platform_table_begin") + len("!platform_table_begin\n")
    end = txt.index("!platform_table_end")
    a = pd.read_csv(io.StringIO(txt[start:end]), sep="\t", dtype=str)
    sym = "Gene symbol" if "Gene symbol" in a.columns else "Gene Symbol"
    a = a[["ID", sym]].dropna()
    a = a[~a[sym].str.contains("///")]  # drop ambiguous multi-gene probes
    return dict(zip(a["ID"], a[sym]))


def read_series(path):
    lines = gzip.open(path, "rt").read().splitlines()
    meta = {}
    for l in lines:
        if l.startswith("!Sample_characteristics_ch1") or l.startswith("!Sample_geo_accession"):
            v = [x.strip('"') for x in l.split("\t")[1:]]
            if l.startswith("!Sample_geo_accession"):
                meta["gsm"] = v
            else:
                key = v[0].split(":")[0].strip()
                meta[key] = [x.split(":", 1)[1].strip() if ":" in x else x for x in v]
    i0 = lines.index("!series_matrix_table_begin") + 1
    i1 = lines.index("!series_matrix_table_end")
    X = pd.read_csv(io.StringIO("\n".join(lines[i0:i1])), sep="\t", index_col=0)
    return X, pd.DataFrame(meta).set_index("gsm")


def to_genes(X, annot):
    X = X.copy()
    X.index = X.index.astype(str)
    X["gene"] = X.index.map(annot)
    X = X.dropna(subset=["gene"])
    if X.drop(columns="gene").max().max() > 100:  # not yet log scale
        num = X.drop(columns="gene").clip(lower=1)
        X = pd.concat([np.log2(num), X["gene"]], axis=1)
    X["mean"] = X.drop(columns="gene").mean(1)
    X = X.sort_values("mean", ascending=False).drop_duplicates("gene")
    return X.set_index("gene").drop(columns="mean").T  # samples x genes


def zscore(df):
    return (df - df.mean()) / (df.std() + 1e-8)


# ---------- load cohorts
E = pd.read_csv(P / "expr.csv.gz", index_col=0)
CNA = pd.read_csv(P / "cna.csv.gz", index_col=0)
METH = pd.read_csv(P / "meth.csv.gz", index_col=0)
CL = pd.read_csv(P / "clinical.csv", index_col=0)
t, e = CL["PFS_months"].values, CL["PFS_event"].values

X1, M1 = read_series(G / "GSE22138.txt.gz")
X1 = to_genes(X1, read_annot(G / "GPL570.annot.gz"))
y1 = pd.DataFrame({"time": pd.to_numeric(M1["months to endpoint"], errors="coerce"),
                   "event": (M1["metastasis"] == "yes").astype(int)}, index=M1.index)

X2, M2 = read_series(G / "GSE44295.txt.gz")
X2 = to_genes(X2, read_annot(G / "GPL6883.annot.gz"))
keep = M2["cell type"] == "primary uveal melanoma cells" if "cell type" in M2 else M2.index == M2.index
M2 = M2[keep & (M2["survival (days)"] != "unknown") & M2["metastasis"].isin(["0", "1"])]
y2 = pd.DataFrame({"time": pd.to_numeric(M2["survival (days)"], errors="coerce") / 30.44,
                   "event": (M2["metastasis"] == "1").astype(int)}, index=M2.index)

cohorts = {}
for name, X, y in [("GSE22138", X1, y1), ("GSE44295", X2, y2)]:
    y = y.dropna()
    X = X.loc[X.index.intersection(y.index)]
    cohorts[name] = (X, y.loc[X.index])
    print(name, X.shape, "events", int(y.loc[X.index, "event"].sum()))

universe = E.columns.intersection(X1.columns).intersection(X2.columns)
print("shared genes:", len(universe))
Eu = E[universe]
Ez = zscore(Eu)

# ---------- train frozen models on all TCGA
models = {}
zs = np.abs(univariate_cox_z(Ez.values, t, e))
top = np.argsort(-zs)[:10]
models["Univariate top-10 + ridge Cox"] = (list(universe[top]), ridge_cox(Ez.values[:, top], t, e, 1.0))
for name, l1 in [("Cox-LASSO", 1.0), ("Cox elastic net", 0.5)]:
    coef = coxnet_cv(Ez.values, t, e, l1)
    nz = coef != 0
    models[name] = (list(universe[nz]), coef[nz])
cand = np.argsort(-zs)[:300]
a = anchoring_score(Eu.iloc[:, cand], CNA, METH).values
pr = (zs[cand] - zs[cand].mean()) / zs[cand].std() + (a - a.mean()) / (a.std() + 1e-8)
pf = np.exp(-pr / 2); pf = pf / pf.mean()
coef = coxnet_cv(Ez.values[:, cand], t, e, 1.0, penalty_factor=pf)
nz = coef != 0
models["Anchored adaptive LASSO"] = (list(universe[cand][nz]), coef[nz])

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
c5 = np.argsort(-zs)[:500]
rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                           n_jobs=1, random_state=0).fit(Ez.values[:, c5], Surv.from_arrays(e.astype(bool), t))
RSF_GENES = list(universe[c5])

# ---- controls and comparators (same TCGA training data, same shared genes)
import re
band = {l.split("\t")[0]: l.split("\t")[2].strip() for l in open(os.environ["CNA_TXT"]) if not l.startswith("Hugo")}
def on_risk_arm(g):
    mm = re.match(r"^(\d+)([pq])", band.get(g, ""))
    return bool(mm) and (mm.group(1) == "3" or (mm.group(1), mm.group(2)) in [("8", "q"), ("1", "p")])
zc = (zs[cand] - zs[cand].mean()) / zs[cand].std()
apos = np.array([on_risk_arm(g) for g in universe[cand]], float)
for nm, prv in [("Adaptive LASSO, no anchoring", zc),
                ("Adaptive LASSO, positional prior", zc + (apos - apos.mean()) / apos.std())]:
    pfv = np.exp(-prv / 2); pfv = pfv / pfv.mean()
    cf = coxnet_cv(Ez.values[:, cand], t, e, 1.0, penalty_factor=pfv)
    models[nm] = (list(universe[cand][cf != 0]), cf[cf != 0])
GEP = ["HTR2B", "ID2", "MTUS1", "ECM1", "ROBO1", "SATB1", "LTA4H", "EIF1B", "FXR1", "CDH1", "LMCD1", "RAB31"]
for nm, gl in [("15-gene-class proxy", GEP), ("15-gene-class proxy + PRAME", GEP + ["PRAME"])]:
    gl = [g for g in gl if g in universe]
    models[nm] = (gl, ridge_cox(Ez[gl].values, t, e, 1.0))
from rl_agent2 import OASIS2
o2 = OASIS2(reward="boot", seed=0).fit(Ez, t, e, CNA, METH)   # fitted on z-scored TCGA -> same scale as z-scored GEO
models["OASIS-2 agent (single signature)"] = (o2.selected_, o2.beta_)
CALLABLES = {"OASIS-2 agent (policy ensemble)": (lambda Xz: o2.predict_risk_ensemble(Xz[universe]), len(o2.ens_genes_)),
             "Random survival forest": (lambda Xz: rsf.predict(Xz[RSF_GENES].values), 500)}

oasis_runs = []
for seed in range(10):
    m = OASIS(seed=seed).fit(Eu, t, e, CNA, METH)
    oasis_runs.append(m)
    models[f"OASIS seed {seed}"] = (m.selected_, m.beta_)

# consensus signature: genes chosen in >= 50% of seeds, refit once
from collections import Counter
cnt = Counter(g for m in oasis_runs for g in m.selected_)
consensus = [g for g, c in cnt.items() if c >= 5]
b = ridge_cox(Ez[consensus].values, t, e, 1.0)
models["OASIS agent (original, consensus)"] = (consensus, b)

# OASIS+ was pre-specified to go to external validation only if it beat OASIS in TCGA CV.
# It did not (0.752 vs 0.756), so it is OFF for the reported results. Set RUN_PLUS=1 to run it anyway.
import os
RUN_PLUS = os.environ.get("RUN_PLUS") == "1"
PLUS_MAIN = None
if RUN_PLUS:
    from rl_agent import OASISPlus
    PLUS_MAIN = OASISPlus(n_agents=10, shift_sigma=0.3, seed=0).fit(Eu, t, e, CNA, METH)
    models["OASIS+ (final, 10 agents)"] = (PLUS_MAIN.selected_, PLUS_MAIN.beta_)
    for seed in range(1, 6):
        m = OASISPlus(n_agents=5, shift_sigma=0.3, seed=seed).fit(Eu, t, e, CNA, METH)
        models[f"OASIS+ seed {seed}"] = (m.selected_, m.beta_)

# ---------- evaluate
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv as _Surv


def surv_at(train_score, test_score, horizon=36.0):
    """Recalibrate a score on TCGA with a 1-D Cox model + Breslow baseline; return S(horizon | x)."""
    sd = train_score.std() + 1e-8
    cph = CoxPHSurvivalAnalysis(alpha=1e-4).fit((train_score / sd)[:, None], _Surv.from_arrays(e.astype(bool), t))
    fns = cph.predict_survival_function((test_score / sd)[:, None])
    return np.array([fn(min(horizon, fn.x[-1])) for fn in fns])


def score_fn(mname):
    if mname in CALLABLES:
        return CALLABLES[mname][0], CALLABLES[mname][1]
    genes, beta = models[mname]
    return (lambda Xz, g=genes, b=np.asarray(beta): Xz[g].values @ b if len(g) else np.zeros(len(Xz))), len(genes)


rows, risks = [], []
names = list(CALLABLES) + list(models)
for cname, (X, y) in cohorts.items():
    Xz = zscore(X)
    for mname in names:
        fn, ng = score_fn(mname)
        r = fn(Xz)
        c = cindex(r, y["time"].values, y["event"].values) if np.std(r) > 0 else 0.5
        rows.append(dict(cohort=cname, method=mname, n_genes=ng, cindex=c))
        if "seed" not in mname:
            s36 = surv_at(fn(Ez), r) if np.std(r) > 0 else np.full(len(r), np.nan)
            risks.append(pd.DataFrame({"cohort": cname, "method": mname, "sample": y.index, "risk": r,
                                       "surv36": s36, "time": y["time"].values, "event": y["event"].values}))
pd.concat(risks).to_csv(OUT / "external_risk_scores.csv", index=False)
json.dump({k: {"genes": list(g), "beta": list(map(float, b))} for k, (g, b) in models.items()
           if "seed" not in k}, open(OUT / "fixed_models.json", "w"), indent=1)
res = pd.DataFrame(rows)
seeds = res[res.method.str.startswith("OASIS seed")].groupby("cohort")["cindex"].agg(["mean", "std"])
pseeds = res[res.method.str.startswith("OASIS+ seed")].groupby("cohort")["cindex"].agg(["mean", "std"])
res.to_csv(OUT / "external_all.csv", index=False)
tab = res[~res.method.str.contains("seed")].pivot(index="method", columns="cohort", values="cindex")
tab.loc["OASIS (mean of 10 seeds)"] = seeds["mean"]
if RUN_PLUS:
    tab.loc["OASIS+ (mean of 5 seeds)"] = pseeds["mean"]
    print("OASIS+ seed sd:", pseeds["std"].round(3).to_dict(), "final genes:", PLUS_MAIN.selected_)
tab.to_csv(OUT / "external_summary.csv")
json.dump({"consensus_signature": consensus, "counts": dict(cnt),
           "oasis_seed_sd": seeds["std"].to_dict()}, open(OUT / "rl_consensus_signature.json", "w"), indent=1)
print(tab.round(3))
print("seed sd:", seeds["std"].round(3).to_dict())
print("consensus:", consensus)
