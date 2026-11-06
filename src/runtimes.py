"""Wall-clock time of one full-TCGA fit per method (2-core CPU; one core in use).
Usage: python runtimes.py PROCESSED_DIR RESULTS_DIR"""
import sys, time, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); warnings.filterwarnings("ignore")
from rl_agent import univariate_cox_z, anchoring_score, OASIS
from rl_agent2 import OASIS2
from run_cv_helpers import coxnet_cv
P, R = Path(sys.argv[1]), Path(sys.argv[2])
E = pd.read_csv(P / "expr.csv.gz", index_col=0); CNA = pd.read_csv(P / "cna.csv.gz", index_col=0)
METH = pd.read_csv(P / "meth.csv.gz", index_col=0); CL = pd.read_csv(P / "clinical.csv", index_col=0).loc[E.index]
t, e = CL.PFS_months.values, CL.PFS_event.values
Ez = (E - E.mean()) / E.std()
out = {}
t0 = time.time()
zs = np.abs(univariate_cox_z(Ez.values, t, e)); cand = np.argsort(-zs)[:300]
a = anchoring_score(E.iloc[:, cand], CNA, METH).values
pr = (zs[cand] - zs[cand].mean()) / zs[cand].std() + (a - a.mean()) / (a.std() + 1e-8)
pf = np.exp(-pr / 2); pf /= pf.mean()
coxnet_cv(Ez.values[:, cand], t, e, 1.0, penalty_factor=pf)
out["Anchored adaptive LASSO (screen + anchoring + inner CV)"] = time.time() - t0
t0 = time.time(); coxnet_cv(Ez.values, t, e, 1.0); out["Cox-LASSO, all genes (inner CV)"] = time.time() - t0
t0 = time.time(); OASIS(seed=0).fit(E, t, e, CNA, METH); out["RL selector, original"] = time.time() - t0
t0 = time.time(); OASIS2(reward="boot", seed=0).fit(Ez, t, e, CNA, METH); out["RL selector, fresh splits + policy ensemble"] = time.time() - t0
json.dump(out, open(R / "runtimes.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k}: {v:.1f} s")
