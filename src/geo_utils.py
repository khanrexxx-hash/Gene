"""GEO helpers shared by external.py and pooled.py."""
import gzip, io
import numpy as np
import pandas as pd


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


