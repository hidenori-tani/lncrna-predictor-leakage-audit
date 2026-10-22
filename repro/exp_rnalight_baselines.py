#!/usr/bin/env python
"""Target A / RNAlight benchmark -- several model families on the SAME task and data, through the
same protocol changes the R1 analysis applied to RNAlight alone.

Reviewer 2 (R1 round), point 1: "... compare several baseline models on the same task and dataset."

Models (all on the pooled, de-duplicated shipped train + test transcripts, as in R1):
  length_lr       logistic regression on log10 length                         (1 feature)
  gc_lr           logistic regression on GC fraction                          (1 feature)
  length_gc_lr    both                                                       (2 features)
  dinuc_lr        logistic regression on 16 dinucleotide fractions
  kmer_lr         logistic regression on the 1,344 k-mer 3/4/5 frequencies RNAlight uses
  kmer_knn        25-nearest-neighbour vote on the same k-mer profiles (cosine): the family most
                  able to exploit near-duplicate or homologous transcripts across a split
  kmer_lgbm       LightGBM on the same 1,344 features = RNAlight's model, as run in R1

Protocols (identical for every model):
  published       the shipped train / test split, scored on the shipped test set
  cv              StratifiedKFold(5) repeated over 5 seeds (100 = the R1 seed, 0, 1, 7, 13)
  homology_50/80  GroupKFold(5) over the R1 MMseqs2 linclust clusters at 50 % and 80 % identity
  matched         StratifiedKFold(5), seed 100, on the R1 length x GC matched subset (n = 2,764)
  resplit         200 random train/test splits with the shipped test size and class counts, to
                  place the published single-split value inside the distribution of single splits
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "external", "RNAlight")
BASE = os.path.join(REPO, "lncRNA", "03_Model_Construction")
SEED = 100
CV_SEEDS = [100, 0, 1, 7, 13]
N_JOBS = 4
KMERS = ["".join(t) for k in (3, 4, 5) for t in itertools.product("ACGT", repeat=k)]
DINUC = ["".join(t) for t in itertools.product("ACGT", repeat=2)]


def load_pool():
    tr = pd.read_csv(os.path.join(BASE, "lncRNA_sublocation_TrainingSet.tsv"), sep="\t")
    te = pd.read_csv(os.path.join(BASE, "lncRNA_sublocation_TestSet.tsv"), sep="\t")
    tr["orig"], te["orig"] = "train", "test"
    pool = pd.concat([tr, te], ignore_index=True).drop_duplicates("ensembl_transcript_id").reset_index(drop=True)
    return pool


def featurize(seqs):
    """R1's featurize: k-mer 3/4/5 counts by str.count, normalised to sum 1 per transcript."""
    raw = np.zeros((len(seqs), len(KMERS)))
    for j, m in enumerate(KMERS):
        raw[:, j] = [s.count(m) for s in seqs]
    return raw / raw.sum(axis=1, keepdims=True)


def dinuc_frac(seqs):
    out = np.zeros((len(seqs), 16))
    for r, s in enumerate(seqs):
        tot = max(len(s) - 1, 1)
        for i in range(len(s) - 1):
            w = s[i:i + 2]
            if w in DINUC:
                out[r, DINUC.index(w)] += 1
        out[r] /= tot
    return out


def models(seed):
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    return {
        "length_lr": ("len", lr()),
        "gc_lr": ("gc", lr()),
        "length_gc_lr": ("lengc", lr()),
        "dinuc_lr": ("di", lr()),
        "kmer_lr": ("kmer", make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))),
        "kmer_knn": ("kmer", KNeighborsClassifier(n_neighbors=25, metric="cosine", n_jobs=N_JOBS)),
        "kmer_lgbm": ("kmer", lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=N_JOBS,
                                                 verbose=-1)),
    }


def score(splits, F, y, seed, names=None):
    out = {}
    for name, (fkey, _) in models(seed).items():
        if names and name not in names:
            continue
        aucs = []
        for tr, te in splits:
            m = models(seed)[name][1]
            m.fit(F[fkey][tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(F[fkey][te])[:, 1]))
        out[name] = aucs
    return out


def clusters(ids, idv):
    rep = {}
    for line in open(os.path.join(HERE, "mmseqs_out", f"lin_{idv}_cluster.tsv")):
        r, m = line.rstrip("\n").split("\t")
        rep[m] = r
    return np.array([rep[i] for i in ids])


def matched_subset(L, GC, y):
    """R1's length/GC matching (exp_confound_matched.py): deciles of log10 length x GC, equal
    numbers of nuclear and cytoplasmic transcripts per cell, rng seed 0."""
    lb = pd.qcut(np.log10(L), 10, labels=False, duplicates="drop")
    gb = pd.qcut(GC, 10, labels=False, duplicates="drop")
    rng = np.random.default_rng(0)
    keep = []
    df = pd.DataFrame({"i": np.arange(len(y)), "y": y, "lb": lb, "gb": gb})
    for (_, _), g in df.groupby(["lb", "gb"]):
        n1, n0 = g[g.y == 1], g[g.y == 0]
        k = min(len(n1), len(n0))
        if k == 0:
            continue
        keep += list(rng.choice(n1.i.values, k, replace=False))
        keep += list(rng.choice(n0.i.values, k, replace=False))
    return np.array(sorted(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "rnalight_baselines_results.json"))
    ap.add_argument("--resplits", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    pool = load_pool()
    seqs = pool["cdna"].str.upper().tolist()
    y = pool["tag"].values.astype(int)
    ids = pool["ensembl_transcript_id"].values.astype(str)
    L = np.array([len(s) for s in seqs], float)
    GC = np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs])
    K = featurize(seqs)
    F = {"len": np.log10(L)[:, None], "gc": GC[:, None], "lengc": np.c_[np.log10(L), GC],
         "di": dinuc_frac(seqs), "kmer": K}
    print(f"pool {len(y)} (nuclear {int(y.sum())}); features ready ({time.time()-t0:.0f}s)", flush=True)
    res = {"n": int(len(y)), "n_positive": int(y.sum())}

    # published single split
    tr = np.where(pool["orig"].values == "train")[0]
    te = np.where(pool["orig"].values == "test")[0]
    res["published"] = {k: v[0] for k, v in score([(tr, te)], F, y, SEED).items()}
    res["published"]["n_test"] = int(len(te))
    print("published:", {k: round(v, 4) for k, v in res["published"].items() if k != "n_test"}, flush=True)

    # repeated CV
    cv = {}
    for s in CV_SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        for name, aucs in score(list(skf.split(K, y)), F, y, s).items():
            cv.setdefault(name, []).append(float(np.mean(aucs)))
    res["cv"] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), values=v) for k, v in cv.items()}
    print("cv:", {k: round(v["mean"], 4) for k, v in res["cv"].items()}, flush=True)

    # homology-aware
    for idv in ("05", "08"):
        g = clusters(ids, idv)
        sp = list(GroupKFold(n_splits=5).split(K, y, groups=g))
        res[f"homology_{idv}"] = {k: dict(mean=float(np.mean(v)), folds=[float(x) for x in v])
                                  for k, v in score(sp, F, y, SEED).items()}
        res[f"homology_{idv}"]["n_clusters"] = int(len(set(g)))
        print(f"homology {idv}:", {k: round(v["mean"], 4) for k, v in res[f'homology_{idv}'].items()
                                    if isinstance(v, dict)}, flush=True)
    # random-split reference with the R1 seed, for paired deltas against the homology rows
    sp = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(K, y))
    res["cv_seed100_folds"] = {k: [float(x) for x in v] for k, v in score(sp, F, y, SEED).items()}

    # matched subset
    keep = matched_subset(L, GC, y)
    Fm = {k: v[keep] for k, v in F.items()}
    ym = y[keep]
    sp = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Fm["kmer"], ym))
    res["matched"] = {k: float(np.mean(v)) for k, v in score(sp, Fm, ym, SEED).items()}
    res["matched"]["n"] = int(len(keep))
    print("matched:", {k: round(v, 4) for k, v in res["matched"].items() if k != "n"}, flush=True)

    # where the published single split sits among random single splits with the shipped test size
    # AND the shipped test class counts (StratifiedShuffleSplit would keep the pooled prevalence,
    # which gives about 199 positives in 380 rather than the shipped 196 -- Codex review, 2026-10-22)
    n_pos_te = int(y[te].sum())
    n_neg_te = int(len(te) - n_pos_te)
    pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rs = np.random.default_rng(SEED)
    splits = []
    for _ in range(args.resplits):
        te_r = np.r_[rs.choice(pos_idx, n_pos_te, replace=False), rs.choice(neg_idx, n_neg_te, replace=False)]
        tr_r = np.setdiff1d(np.arange(len(y)), te_r)
        splits.append((tr_r, np.sort(te_r)))
    assert all(int(y[t].sum()) == n_pos_te and len(t) == len(te) for _, t in splits)
    res["resplit_design"] = dict(n_test=int(len(te)), n_pos_test=n_pos_te, n_neg_test=n_neg_te)
    dist = score(splits, F, y, SEED, names=("kmer_lgbm", "kmer_lr", "length_gc_lr"))
    res["resplit"] = {}
    for k, v in dist.items():
        v = np.array(v)
        pub = res["published"][k]
        res["resplit"][k] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                 p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                                 published=float(pub), percentile_of_published=float((v < pub).mean() * 100),
                                 values=[float(x) for x in v])
    print("resplit:", {k: (round(v["mean"], 4), round(v["percentile_of_published"], 1))
                       for k, v in res["resplit"].items()}, flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"done ({time.time()-t0:.0f}s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
