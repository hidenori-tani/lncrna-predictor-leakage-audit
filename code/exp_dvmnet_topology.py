#!/usr/bin/env python
"""Target B / DVMnet — protocol leakage audit (model-agnostic).

DVMnet evaluates lncRNA-miRNA interaction prediction with KFold(5, shuffle,
random_state=42) over EDGES (a random-edge split of a bipartite graph). This
leaks node degree/identity: held-out positive edges connect nodes that are still
heavily present in the training graph, so trivial popularity heuristics score
high WITHOUT learning any biology.

We show, with NO neural model, that:
  (1) under DVMnet's random-edge split, preferential-attachment (degree product)
      and other topology-only heuristics already achieve high AUROC;
  (2) under an entity-disjoint COLD-START split (held-out nodes never seen in
      training), the same heuristics collapse toward 0.5.
This bounds how much of any model's reported AUROC is a degree/popularity shortcut.
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
rng = np.random.default_rng(SEED)

nl = pd.read_csv(f"{BASE}/node_link.csv")
edges = [(int(a), int(b)) for a, b in zip(nl.node1, nl.node2)]   # (lnc 0..283, mi 284..803)
P = set(edges)
LNC = list(range(0, 284)); MI = list(range(284, 804))
print(f"[data] {len(edges)} edges | {len(LNC)} lncRNA x {len(MI)} miRNA (density={len(edges)/(len(LNC)*len(MI)):.4f})")

def sample_negatives(k, exclude, seed):
    r = np.random.default_rng(seed); negs = set()
    while len(negs) < k:
        u = int(r.integers(0, 284)); v = int(r.integers(284, 804))
        if (u, v) not in exclude and (u, v) not in negs: negs.add((u, v))
    return list(negs)

def degrees(train_edges):
    dl = {u: 0 for u in LNC}; dm = {v: 0 for v in MI}
    for u, v in train_edges: dl[u] += 1; dm[v] += 1
    return dl, dm

def neighbors(train_edges):
    Nl = {u: set() for u in LNC}; Nm = {v: set() for v in MI}
    for u, v in train_edges: Nl[u].add(v); Nm[v].add(u)
    return Nl, Nm

def score_edges(pairs, dl, dm, Nl, Nm):
    """returns dict of {heuristic: array of scores for pairs}"""
    PA = np.array([dl[u]*dm[v] for u, v in pairs], float)              # preferential attachment (degree)
    DEGSUM = np.array([dl[u]+dm[v] for u, v in pairs], float)          # degree sum
    # 2-hop resource allocation for bipartite: paths u-w-u'-v weighted by 1/deg
    RA = []
    for u, v in pairs:
        s = 0.0
        for u2 in Nm[v]:               # lncRNAs interacting with v
            common = Nl[u] & Nl[u2]    # shared miRNAs between u and u2
            for w in common:
                if dm[w] > 0: s += 1.0/dm[w]
        RA.append(s)
    return {"pref_attach(degree)": PA, "degree_sum": DEGSUM, "resource_alloc_2hop": RA}

def evaluate(train_edges, val_pos, val_neg):
    dl, dm = degrees(train_edges); Nl, Nm = neighbors(train_edges)
    sp = score_edges(val_pos, dl, dm, Nl, Nm)
    sn = score_edges(val_neg, dl, dm, Nl, Nm)
    y = np.r_[np.ones(len(val_pos)), np.zeros(len(val_neg))]
    out = {}
    for h in sp:
        s = np.r_[np.asarray(sp[h]), np.asarray(sn[h])]
        try: out[h] = roc_auc_score(y, s)
        except Exception: out[h] = float("nan")
    return out

# ---------- (1) RANDOM-EDGE split (DVMnet's protocol) ----------
print("\n=== (1) RANDOM-EDGE KFold(5, shuffle, seed=42) — DVMnet's protocol ===")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
E = np.array(edges)
accum = {}
for k,(tr,va) in enumerate(kf.split(E)):
    train_edges = [tuple(x) for x in E[tr]]; val_pos = [tuple(x) for x in E[va]]
    val_neg = sample_negatives(len(val_pos), P, seed=SEED+k)
    res = evaluate(train_edges, val_pos, val_neg)
    for h,a in res.items(): accum.setdefault(h,[]).append(a)
for h,a in accum.items(): print(f"  {h:<22} AUROC = {np.mean(a):.4f} ± {np.std(a):.4f}")

# ---------- (2) COLD-START (entity-disjoint) split ----------
print("\n=== (2) COLD-START entity-disjoint split (held-out nodes unseen in training) ===")
accum2 = {}
for rep in range(5):
    r = np.random.default_rng(100+rep)
    lnc_hold = set(r.choice(LNC, size=int(0.3*len(LNC)), replace=False).tolist())
    mi_hold  = set(r.choice(MI,  size=int(0.3*len(MI)),  replace=False).tolist())
    train_edges = [(u,v) for u,v in edges if u not in lnc_hold and v not in mi_hold]
    val_pos     = [(u,v) for u,v in edges if u in lnc_hold and v in mi_hold]
    if len(val_pos) < 10: continue
    # negatives: held-out lnc x held-out mi pairs that are not real edges
    negpool = [(u,v) for u in lnc_hold for v in mi_hold if (u,v) not in P]
    idx = r.choice(len(negpool), size=min(len(val_pos), len(negpool)), replace=False)
    val_neg = [negpool[i] for i in idx]
    res = evaluate(train_edges, val_pos, val_neg)
    for h,a in res.items(): accum2.setdefault(h,[]).append(a)
    if rep==0: print(f"  (fold0: train_edges={len(train_edges)} val_pos={len(val_pos)} val_neg={len(val_neg)})")
for h,a in accum2.items(): print(f"  {h:<22} AUROC = {np.mean(a):.4f} ± {np.std(a):.4f}")

print("\n[interpretation] High AUROC under (1) from pure degree/topology = the random-edge protocol's")
print("leakage surface. Collapse toward 0.50 under (2) = that 'performance' was popularity, not interaction biology.")
