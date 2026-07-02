#!/usr/bin/env python
"""Target A / RNAlight — composition-confound baselines.
How much of RNAlight's localization AUROC is explained by trivial sequence
properties (length, GC, mono/di-nucleotide composition) vs the full k-mer model?
All under the same random StratifiedKFold(5), LightGBM, SEED=100.
"""
import itertools, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

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
SEED = 100
tr = pd.read_csv(f"{BASE}/lncRNA_sublocation_TrainingSet.tsv", sep="\t")
te = pd.read_csv(f"{BASE}/lncRNA_sublocation_TestSet.tsv", sep="\t")
pool = pd.concat([tr, te], ignore_index=True).drop_duplicates("ensembl_transcript_id").reset_index(drop=True)
seqs = pool["cdna"].tolist(); y = pool["tag"].values.astype(int)
print(f"[data] n={len(seqs)} pos={y.sum()} neg={(y==0).sum()}")

def kmer_freq(seqs, ks):
    mers = [ "".join(t) for k in ks for t in itertools.product("ACGT", repeat=k) ]
    M = np.zeros((len(seqs), len(mers)))
    for j,m in enumerate(mers):
        M[:,j] = [s.count(m) for s in seqs]
    return M / M.sum(axis=1, keepdims=True)

L = np.array([len(s) for s in seqs], dtype=float)
GC = np.array([(s.count("G")+s.count("C"))/len(s) for s in seqs])
feats = {
    "length_only (log-len)":      np.log10(L).reshape(-1,1),
    "GC_only":                    GC.reshape(-1,1),
    "length+GC":                  np.column_stack([np.log10(L), GC]),
    "mononucleotide (4)":         kmer_freq(seqs,[1]),
    "dinucleotide (16)":          kmer_freq(seqs,[2]),
    "3-mer (64)":                 kmer_freq(seqs,[3]),
    "k-mer 3/4/5 FULL (1344)":    kmer_freq(seqs,[3,4,5]),
}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds = list(skf.split(seqs, y))
def auroc(X):
    a=[]
    for tr_i,te_i in folds:
        m=lgb.LGBMClassifier(objective="binary",random_state=SEED,n_jobs=-1,verbose=-1).fit(X[tr_i],y[tr_i])
        a.append(roc_auc_score(y[te_i], m.predict_proba(X[te_i])[:,1]))
    return np.mean(a), np.std(a)

print(f"\n{'feature set':<28}{'AUROC (5-fold CV)':>20}")
for name,X in feats.items():
    mu,sd = auroc(np.asarray(X))
    print(f"  {name:<26}{mu:6.4f} ± {sd:.4f}")

# label vs length/GC association (are the labels themselves confounded?)
from scipy.stats import mannwhitneyu
print("\n[label confound] nuclear(1) vs cytoplasmic(0):")
print(f"  median length: nuc={np.median(L[y==1]):.0f}  cyto={np.median(L[y==0]):.0f}  MWU p={mannwhitneyu(L[y==1],L[y==0]).pvalue:.2e}")
print(f"  median GC    : nuc={np.median(GC[y==1]):.3f}  cyto={np.median(GC[y==0]):.3f}  MWU p={mannwhitneyu(GC[y==1],GC[y==0]).pvalue:.2e}")
