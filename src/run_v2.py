"""OASIS-2 on the same 10x5-fold splits. Development = repeats 0-4, confirmation = 5-9.

Usage: python run_v2.py PROCESSED_DIR OUT_DIR REP_START REP_END [variant ...]
Variants: boot (fresh bootstrap reward), cv (fresh inner-CV reward).
Each fit yields two predictors: single signature and policy ensemble.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from rl_agent import cindex
from rl_agent2 import OASIS2

warnings.filterwarnings("ignore")
P, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R0, R1 = int(sys.argv[3]), int(sys.argv[4])
VARS = sys.argv[5:] or ["boot", "cv"]
CK = OUT / "v2_folds"; CK.mkdir(parents=True, exist_ok=True)

E = pd.read_csv(P / "expr.csv.gz", index_col=0)
CNA = pd.read_csv(P / "cna.csv.gz", index_col=0)
METH = pd.read_csv(P / "meth.csv.gz", index_col=0)
CL = pd.read_csv(P / "clinical.csv", index_col=0)
T, EV = CL["PFS_months"].values, CL["PFS_event"].values


def run_fold(rep, k, tr, te, var):
    f = CK / f"{var}_{rep}_{k}.json"
    if f.exists():
        return json.load(open(f))
    m = OASIS2(reward=var, seed=1000 * rep + k).fit(E.iloc[tr], T[tr], EV[tr], CNA.iloc[tr], METH.iloc[tr])
    o = dict(rep=rep, fold=k, var=var,
             c_single=cindex(m.predict_risk(E.iloc[te]), T[te], EV[te]),
             c_ens=cindex(m.predict_risk_ensemble(E.iloc[te]), T[te], EV[te]),
             sel=m.selected_, n_ens_genes=len(m.ens_genes_), history=m.history_)
    json.dump(o, open(f, "w"))
    return o


jobs = []
for rep in range(R0, R1 + 1):
    skf = StratifiedKFold(5, shuffle=True, random_state=rep)
    for k, (tr, te) in enumerate(skf.split(E.values, EV)):
        for v in VARS:
            jobs.append((rep, k, tr, te, v))
out = Parallel(n_jobs=2, verbose=2)(delayed(run_fold)(*j) for j in jobs)
print("done", len(out))
