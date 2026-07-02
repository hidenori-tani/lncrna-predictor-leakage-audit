#!/usr/bin/env python
"""Target A / RNAlight — study-effort (famous-few) analysis (Pillar 1).

Uses PubMed co-occurrence counts per lncRNA gene symbol (repro/pubmed_counts.tsv)
as a study-effort proxy. Tests whether RNAlight's apparent performance is carried
by well-studied lncRNAs and fails to generalize to the obscure long tail.
"""
import itertools, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
CNT  = _os.path.join(DATA, "pubmed_counts.tsv")
SEED = 100

tr = pd.read_csv(f"{BASE}/lncRNA_sublocation_TrainingSet.tsv", sep="\t")
te = pd.read_csv(f"{BASE}/lncRNA_sublocation_TestSet.tsv", sep="\t")
pool = pd.concat([tr, te], ignore_index=True).drop_duplicates("ensembl_transcript_id").reset_index(drop=True)

cnt = pd.read_csv(CNT, sep="\t", header=None, names=["name","pubmed"])
cnt = cnt[cnt.pubmed >= 0]                       # drop error rows (-1)
cmap = dict(zip(cnt.name, cnt.pubmed))
pool["pubmed"] = pool["name"].map(cmap)
have = pool["pubmed"].notna().sum()
print(f"[coverage] pubmed counts mapped for {have}/{len(pool)} pooled lncRNAs")
pool = pool.dropna(subset=["pubmed"]).reset_index(drop=True)
pool["pubmed"] = pool["pubmed"].astype(int)

# featurize
NUC = ["A","C","G","T"]; KMERS = ["".join(t) for k in (3,4,5) for t in itertools.product(NUC, repeat=k)]
def featurize(seqs):
    raw = np.zeros((len(seqs), len(KMERS)))
    for j,m in enumerate(KMERS): raw[:,j] = [s.count(m) for s in seqs]
    return raw/raw.sum(axis=1, keepdims=True)
X = featurize(pool["cdna"].tolist()); y = pool["tag"].values.astype(int)
eff = pool["pubmed"].values

print(f"[study-effort] pubmed: median={np.median(eff):.0f} mean={eff.mean():.1f} zero-count={np.mean(eff==0)*100:.0f}% max={eff.max()}")
# label vs effort confound
from scipy.stats import mannwhitneyu
print(f"[confound] pubmed nuclear vs cyto: median {np.median(eff[y==1]):.0f} vs {np.median(eff[y==0]):.0f}  MWU p={mannwhitneyu(eff[y==1],eff[y==0]).pvalue:.2e}")

def lgbm(): return lgb.LGBMClassifier(objective="binary", random_state=SEED, n_jobs=-1, verbose=-1)

print(f"[strata] unstudied(pubmed==0): n={int((eff==0).sum())} ({100*np.mean(eff==0):.0f}%) | studied(>=1): n={int((eff>=1).sum())}")
print(f"[confound] fraction studied(>=1): nuclear={np.mean(eff[y==1]>=1):.3f} vs cyto={np.mean(eff[y==0]>=1):.3f}")

# ---- Analysis 1: AUROC by study-effort stratum (cross-validated predictions) ----
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
proba = cross_val_predict(lgbm(), X, y, cv=skf, method="predict_proba")[:,1]
def stratum(e):
    if e == 0: return "0  unstudied"
    if e <= 5: return "1-5 low"
    if e <= 50: return "6-50 medium"
    return ">50 famous"
strat = np.array([stratum(e) for e in eff])
print("\n(1) AUROC by study-effort stratum (5-fold CV predictions):")
for lab in ["0  unstudied","1-5 low","6-50 medium",">50 famous"]:
    m = strat == lab
    if m.sum() > 10 and len(np.unique(y[m])) == 2:
        print(f"   {lab:<14} n={m.sum():4d}  AUROC={roc_auc_score(y[m], proba[m]):.4f}")

# ---- Analysis 2: train on STUDIED, test on the UNSTUDIED majority (the Opinion's core claim) ----
studied = eff >= 1; unstudied = eff == 0
m = lgbm().fit(X[studied], y[studied])
auc_unstudied = roc_auc_score(y[unstudied], m.predict_proba(X[unstudied])[:,1])
# matched control: random train of same size (n studied), test on the rest
rng = np.random.default_rng(SEED); perm = rng.permutation(len(eff))
ntr = int(studied.sum()); rtr, rte = perm[:ntr], perm[ntr:]
m2 = lgbm().fit(X[rtr], y[rtr]); auc_rand = roc_auc_score(y[rte], m2.predict_proba(X[rte])[:,1])
print(f"\n(2) train on STUDIED (pubmed>=1, n={int(studied.sum())}), test on UNSTUDIED majority (n={int(unstudied.sum())}): AUROC={auc_unstudied:.4f}")
print(f"    matched control: random train (n={ntr}), test on rest: AUROC={auc_rand:.4f}")
print(f"    generalization gap famous->uncharacterized: {auc_unstudied-auc_rand:+.4f}")
print("\n[interpretation] AUROC by stratum + the studied->unstudied gap quantify the famous-few bias (Pillar 1):")
print("performance concentrated on well-studied lncRNAs and/or failing on the zero-literature majority.")
