#!/usr/bin/env python
"""Target B / DVMnet — negative-sampling leakage (canonical, reproducible).

Under DVMnet's random-edge KFold(5, seed=42), compare the zero-parameter degree-sum
link-prediction AUROC with (i) uniform-random negatives (DVMnet's actual protocol)
vs (ii) degree-matched negatives (endpoints drawn in proportion to global node degree).
This isolates negative sampling as a second, compounding leakage axis.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# --- path config: edit these or set env vars. Defaults assume repo layout code/ + data/ ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
DATA = _os.environ.get('AUDIT_DATA', _os.path.join(_ROOT, 'data'))
FIG  = _os.environ.get('AUDIT_FIGS', _os.path.join(_ROOT, 'figures'))
RNALIGHT_DIR = _os.environ.get('RNALIGHT_DIR', _os.path.join(_ROOT, 'external', 'RNAlight'))
DVMNET_DIR   = _os.environ.get('DVMNET_DIR',   _os.path.join(_ROOT, 'external', 'DVMnet'))
# --- end path config ---
BASE = _os.path.join(DVMNET_DIR, "dataset")
SEED = 42
nl = pd.read_csv(f"{BASE}/node_link.csv")
edges = [(int(a), int(b)) for a, b in zip(nl.node1, nl.node2)]
P = set(edges); LNC = list(range(0, 284)); MI = list(range(284, 804))

# global degree -> sampling weights for degree-matched negatives
dlA = {u: 0 for u in LNC}; dmA = {v: 0 for v in MI}
for u, v in edges: dlA[u] += 1; dmA[v] += 1
lnc_arr = np.array(LNC); mi_arr = np.array(MI)
pl = np.array([dlA[u] for u in LNC], float); pl /= pl.sum()
pm = np.array([dmA[v] for v in MI], float); pm /= pm.sum()

def uniform_negs(k, seed):
    r = np.random.default_rng(seed); negs = set()
    while len(negs) < k:
        u = int(r.integers(0, 284)); v = int(r.integers(284, 804))
        if (u, v) not in P and (u, v) not in negs: negs.add((u, v))
    return list(negs)

def degree_matched_negs(k, seed):
    r = np.random.default_rng(seed); negs = set()
    while len(negs) < k:
        u = int(r.choice(lnc_arr, p=pl)); v = int(r.choice(mi_arr, p=pm))
        if (u, v) not in P and (u, v) not in negs: negs.add((u, v))
    return list(negs)

def train_degrees(train_edges):
    dl = {u: 0 for u in LNC}; dm = {v: 0 for v in MI}
    for u, v in train_edges: dl[u] += 1; dm[v] += 1
    return dl, dm

def degsum_auroc(neg_fn):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED); E = np.array(edges); a = []
    for k, (tr, va) in enumerate(kf.split(E)):
        te = [tuple(x) for x in E[tr]]; vp = [tuple(x) for x in E[va]]
        vn = neg_fn(len(vp), SEED + k)
        dl, dm = train_degrees(te)
        sp = [dl[u] + dm[v] for u, v in vp]; sn = [dl[u] + dm[v] for u, v in vn]
        y = np.r_[np.ones(len(vp)), np.zeros(len(vn))]; s = np.r_[sp, sn]
        a.append(roc_auc_score(y, s))
    return np.array(a)

u = degsum_auroc(uniform_negs); d = degsum_auroc(degree_matched_negs)
print(f"degree_sum, random-edge + UNIFORM negatives (DVMnet protocol): AUROC {u.mean():.4f} ± {u.std():.4f}")
print(f"degree_sum, random-edge + DEGREE-MATCHED negatives (fair)    : AUROC {d.mean():.4f} ± {d.std():.4f}")
print(f"drop from fixing negatives only: {d.mean()-u.mean():+.4f}")
