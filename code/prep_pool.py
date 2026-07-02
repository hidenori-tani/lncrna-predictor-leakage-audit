#!/usr/bin/env python
"""Prep for the homology-split experiment (Target A = RNAlight).
Pools shipped train+test lncRNAs, writes a FASTA (for mmseqs clustering), and
caches the k-mer 3/4/5 frequency features + labels + ids to an .npz.
"""
import itertools, numpy as np, pandas as pd

# --- path config: edit these or set env vars. Defaults assume repo layout code/ + data/ ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
DATA = _os.environ.get('AUDIT_DATA', _os.path.join(_ROOT, 'data'))
FIG  = _os.environ.get('AUDIT_FIGS', _os.path.join(_ROOT, 'figures'))
RNALIGHT_DIR = _os.environ.get('RNALIGHT_DIR', _os.path.join(_ROOT, 'external', 'RNAlight'))
DVMNET_DIR   = _os.environ.get('DVMNET_DIR',   _os.path.join(_ROOT, 'external', 'DVMnet'))
# --- end path config ---
REPO = RNALIGHT_DIR
BASE = f"{REPO}/lncRNA/03_Model_Construction"
OUT  = DATA

NUC = ["A","C","G","T"]
KMERS = ["".join(t) for k in (3,4,5) for t in itertools.product(NUC, repeat=k)]

def featurize(cdna_list):
    raw = np.zeros((len(cdna_list), len(KMERS)))
    for j, mer in enumerate(KMERS):
        raw[:, j] = [s.count(mer) for s in cdna_list]
    return raw / raw.sum(axis=1, keepdims=True)

tr = pd.read_csv(f"{BASE}/lncRNA_sublocation_TrainingSet.tsv", sep="\t"); tr["orig"]="train"
te = pd.read_csv(f"{BASE}/lncRNA_sublocation_TestSet.tsv", sep="\t");     te["orig"]="test"
pool = pd.concat([tr, te], ignore_index=True)
# de-dup by transcript id (safety) — keep first
before=len(pool); pool = pool.drop_duplicates("ensembl_transcript_id").reset_index(drop=True)
print(f"pooled={before} -> unique transcripts={len(pool)}  labels={dict(pool.tag.value_counts())}")

# FASTA
with open(f"{OUT}/pool.fasta","w") as f:
    for tid, seq in zip(pool["ensembl_transcript_id"], pool["cdna"]):
        f.write(f">{tid}\n{seq}\n")
print(f"wrote {OUT}/pool.fasta")

# features
X = featurize(pool["cdna"].tolist())
np.savez_compressed(f"{OUT}/pool_features.npz",
                    X=X, y=pool["tag"].values.astype(int),
                    ids=pool["ensembl_transcript_id"].values.astype(str),
                    orig=pool["orig"].values.astype(str),
                    kmers=np.array(KMERS))
print(f"cached features: X={X.shape} -> {OUT}/pool_features.npz")
# also seq length for QC
print("seq length: min/median/max =", int(pool.cdna.str.len().min()), int(pool.cdna.str.len().median()), int(pool.cdna.str.len().max()))
