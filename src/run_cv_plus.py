"""Same 10x5-fold splits as run_cv.py, for the OASIS+ variants.

Usage: python run_cv_plus.py PROCESSED_DIR OUT_DIR [n_repeats]
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from rl_agent import OASIS, OASISPlus, cindex

warnings.filterwarnings("ignore")
P, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
OUT.mkdir(parents=True, exist_ok=True)
E = pd.read_csv(P / "expr.csv.gz", index_col=0)
CNA = pd.read_csv(P / "cna.csv.gz", index_col=0)
METH = pd.read_csv(P / "meth.csv.gz", index_col=0)
CL = pd.read_csv(P / "clinical.csv", index_col=0)
T, EV = CL["PFS_months"].values, CL["PFS_event"].values

VARIANTS = {
    "OASIS+ (multi-agent + shift-robust)": lambda s: OASISPlus(n_agents=5, shift_sigma=0.3, seed=s),
    "OASIS+ w/o shift robustness": lambda s: OASISPlus(n_agents=5, shift_sigma=0.0, seed=s),
    "OASIS single agent + shift-robust": lambda s: OASIS(shift_sigma=0.3, seed=s),
}


CK = OUT / "plus_folds"
CK.mkdir(exist_ok=True)


def run_fold(rep, k, tr, te):
    f = CK / f"{rep}_{k}.json"
    if f.exists():  # resume
        return json.load(open(f))
    res, sets, hist = {}, {}, None
    for name, make in VARIANTS.items():
        m = make(1000 * rep + k).fit(E.iloc[tr], T[tr], EV[tr], CNA.iloc[tr], METH.iloc[tr])
        res[name] = cindex(m.predict_risk(E.iloc[te]), T[te], EV[te])
        sets[name] = m.selected_
        if hist is None:
            hist = m.history_
    o = dict(rep=rep, fold=k, cindex=res, sets=sets, history=hist)
    json.dump(o, open(f, "w"))
    return o


folds = []
for rep in range(REPEATS):
    skf = StratifiedKFold(5, shuffle=True, random_state=rep)
    for k, (tr, te) in enumerate(skf.split(E.values, EV)):
        folds.append((rep, k, tr, te))

out = Parallel(n_jobs=2, verbose=5)(delayed(run_fold)(*f) for f in folds)
json.dump(out, open(OUT / "cv_results_plus.json", "w"))
print("done", len(out))
