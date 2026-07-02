#!/usr/bin/env python
"""Target A / RNAlight — R2 controls: (1) multi-seed CV robustness; (2) length/GC
confound-matched evaluation. Addresses Stage-2 reviewers' request for a
confound-controlled test and multi-seed stability.
"""
import itertools, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
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
tr = pd.read_csv(f"{BASE}/lncRNA_sublocation_TrainingSet.tsv", sep="\t")
te = pd.read_csv(f"{BASE}/lncRNA_sublocation_TestSet.tsv", sep="\t")
pool = pd.concat([tr, te], ignore_index=True).drop_duplicates("ensembl_transcript_id").reset_index(drop=True)
seqs = pool["cdna"].tolist(); y = pool["tag"].values.astype(int)
KMERS = ["".join(t) for k in (3,4,5) for t in itertools.product("ACGT", repeat=k)]
def featurize(seqs):
    raw = np.zeros((len(seqs), len(KMERS)))
    for j,m in enumerate(KMERS): raw[:,j] = [s.count(m) for s in seqs]
    return raw/raw.sum(axis=1, keepdims=True)
X = featurize(seqs)
L = np.array([len(s) for s in seqs], float)
GC = np.array([(s.count("G")+s.count("C"))/len(s) for s in seqs])

def lgbm(seed): return lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=-1, verbose=-1)
def cv_auroc(Xm, ym, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); a=[]
    for tri,tei in skf.split(Xm,ym):
        m=lgbm(seed).fit(Xm[tri],ym[tri]); a.append(roc_auc_score(ym[tei], m.predict_proba(Xm[tei])[:,1]))
    return np.mean(a)

# ---- (1) multi-seed CV robustness (full model) ----
seeds = [0,1,7,13,42,100,2024,31337]
means = [cv_auroc(X, y, s) for s in seeds]
print(f"(1) full k-mer model, 5-fold CV across {len(seeds)} seeds: AUROC {np.mean(means):.4f} ± {np.std(means):.4f}  (per-seed {np.round(means,3)})")

# ---- (2) length/GC-matched evaluation ----
# bin log-length x GC into deciles; within each cell keep equal #nuclear/#cyto (subsample majority)
logL = np.log10(L)
lb = pd.qcut(logL, 10, labels=False, duplicates="drop")
gb = pd.qcut(GC, 10, labels=False, duplicates="drop")
rng = np.random.default_rng(0)
keep = []
df = pd.DataFrame({"i":np.arange(len(y)),"y":y,"lb":lb,"gb":gb})
for (_,_), g in df.groupby(["lb","gb"]):
    n1 = g[g.y==1]; n0 = g[g.y==0]; k = min(len(n1), len(n0))
    if k==0: continue
    keep += list(rng.choice(n1.i.values, k, replace=False))
    keep += list(rng.choice(n0.i.values, k, replace=False))
keep = np.array(sorted(keep))
Xm, ym, Lm, GCm = X[keep], y[keep], L[keep], GC[keep]
print(f"\n(2) length/GC-matched set: n={len(keep)} (from {len(y)}); nuclear={int(ym.sum())} cyto={int((ym==0).sum())}")
print(f"    after matching: median length nuc={np.median(Lm[ym==1]):.0f} cyto={np.median(Lm[ym==0]):.0f} MWU p={mannwhitneyu(Lm[ym==1],Lm[ym==0]).pvalue:.2g}")
print(f"                    median GC     nuc={np.median(GCm[ym==1]):.3f} cyto={np.median(GCm[ym==0]):.3f} MWU p={mannwhitneyu(GCm[ym==1],GCm[ym==0]).pvalue:.2g}")
auroc_matched = cv_auroc(Xm, ym, 100)
print(f"    full k-mer model AUROC on length/GC-matched set: {auroc_matched:.4f}  (vs unmatched 0.728)")
print(f"    => drop attributable to length/GC confound: {0.728-auroc_matched:+.4f}")
