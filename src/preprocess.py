"""Preprocess TCGA-UVM PanCancer Atlas data into clean, sample-aligned matrices.

Outputs (in OUT dir):
  expr.csv.gz    log2(RSEM+1), genes filtered, samples x genes
  cna.csv.gz     GISTIC discrete copy number, samples x genes
  meth.csv.gz    beta values (one probe per gene), samples x genes
  clinical.csv    PFS/OS endpoints + clinical markers (chr3, 8q, BAP1, SF3B1, EIF1AX, age, stage)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def to_patient(s):
    return s[:12]


# ---------- expression ----------
e = pd.read_csv(RAW / "data_mrna_seq_v2_rsem.txt", sep="\t")
e = e.dropna(subset=["Hugo_Symbol"]).drop(columns=["Entrez_Gene_Id"])
e = e.groupby("Hugo_Symbol").mean()          # collapse duplicate symbols
e = np.log2(e.clip(lower=0) + 1).T            # samples x genes
e.index = [to_patient(i) for i in e.index]
e = e[~e.index.duplicated()]
# keep genes expressed in >=20% of samples and in the top 70% by variance
expressed = (e > 1).mean() >= 0.2
e = e.loc[:, expressed]
var = e.var()
e = e.loc[:, var >= var.quantile(0.30)]

# ---------- copy number ----------
c = pd.read_csv(RAW / "data_cna.txt", sep="\t")
c = c.dropna(subset=["Hugo_Symbol"]).drop(columns=["Entrez_Gene_Id", "Cytoband"])
c = c.groupby("Hugo_Symbol").mean().T
c.index = [to_patient(i) for i in c.index]
c = c[~c.index.duplicated()]

# ---------- methylation ----------
m = pd.read_csv(RAW / "data_methylation_hm27_hm450_merged.txt", sep="\t")
m = m.dropna(subset=["NAME"]).drop(columns=["ENTITY_STABLE_ID", "DESCRIPTION", "TRANSCRIPT_ID"])
m = m.groupby("NAME").mean().T
m.index = [to_patient(i) for i in m.index]
m = m[~m.index.duplicated()]
m = m.loc[:, m.isna().mean() < 0.2]
m = m.fillna(m.median())

# ---------- clinical ----------
p = pd.read_csv(RAW / "data_clinical_patient.txt", sep="\t", comment="#")
clin = pd.DataFrame(index=p["PATIENT_ID"])
for ep in ["PFS", "OS", "DSS"]:
    clin[f"{ep}_event"] = p[f"{ep}_STATUS"].astype(str).str.startswith("1").astype(int).values
    clin[f"{ep}_months"] = pd.to_numeric(p[f"{ep}_MONTHS"], errors="coerce").values
clin["age"] = pd.to_numeric(p["AGE"], errors="coerce").values
stage = p["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str)
clin["stage_IV"] = stage.str.contains("IV").astype(int).values
clin["stage_III_plus"] = stage.str.contains("III|IV").astype(int).values

arm = pd.read_csv(RAW / "data_armlevel_cna.txt", sep="\t").set_index("NAME").drop(
    columns=["ENTITY_STABLE_ID", "DESCRIPTION"]).T
arm.index = [to_patient(i) for i in arm.index]
clin["chr3_loss"] = (arm.reindex(clin.index)[["3p", "3q"]] == "Loss").any(axis=1).astype(int)
clin["chr8q_gain"] = (arm.reindex(clin.index)["8q"] == "Gain").astype(int)

mut = pd.read_csv(RAW / "data_mutations.txt", sep="\t", low_memory=False)
nonsilent = mut[~mut["Variant_Classification"].isin(["Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "RNA", "IGR"])]
for g in ["BAP1", "SF3B1", "EIF1AX", "GNAQ", "GNA11"]:
    pts = set(nonsilent.loc[nonsilent["Hugo_Symbol"] == g, "Tumor_Sample_Barcode"].map(to_patient))
    clin[f"{g}_mut"] = clin.index.isin(pts).astype(int)

# ---------- align ----------
common = sorted(set(e.index) & set(c.index) & set(m.index) & set(clin.index[clin["PFS_months"].notna()]))
e, c, m, clin = e.loc[common], c.loc[common], m.loc[common], clin.loc[common]
clin.loc[clin["PFS_months"] <= 0, "PFS_months"] = 0.5

e.to_csv(OUT / "expr.csv.gz")
c.to_csv(OUT / "cna.csv.gz")
m.to_csv(OUT / "meth.csv.gz")
clin.to_csv(OUT / "clinical.csv")

print(f"patients aligned: {len(common)}")
print(f"expr {e.shape}  cna {c.shape}  meth {m.shape}")
print(f"PFS events {clin.PFS_event.sum()}  OS events {clin.OS_event.sum()}")
print(clin[["chr3_loss", "chr8q_gain", "BAP1_mut", "SF3B1_mut", "EIF1AX_mut", "GNAQ_mut", "GNA11_mut", "stage_IV"]].sum().to_dict())
