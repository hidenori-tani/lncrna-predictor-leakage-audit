#!/usr/bin/env python
"""Target C / LncPTPred — model-level audit: how much of the reported ">0.9" AUC is composition?

LncPTPred (Brief Bioinform 2025, bbaf432) trains LightGBM on 720 features: 351 from the lncRNA
fragment (k-mer 1-4 counts + RNAfold secondary-structure metrics) and 369 from the protein
(amino-acid physicochemical properties). Their negatives are declared matched to positives on
"number, length, and strands"; composition is NOT among the matched properties.

We rebuild their RNA feature family (k-mer 1-4 counts = 340 features) and represent the protein
side by protein IDENTITY as a LightGBM categorical -- a superset of any fixed physicochemical
lookup, so this is generous to the audited model. RNAfold structure metrics are omitted (they
require ViennaRNA); this is stated as a limitation and makes our reproduction a lower bound.

Conditions
  full            : k-mer 1-4 + protein identity  (their feature family)
  kmer-only       : k-mer 1-4, no protein         (is the protein side doing anything?)
  gc-only         : 1 feature (GC fraction)       (trivial ceiling)
  len-gc-only     : 2 features (length, GC)       (trivial ceiling incl. their imperfect length match)
  full@gc-matched : full features, evaluated on a GC-matched subsample where the composition
                    channel their negative sampling left open has been closed post hoc
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(
    HERE, "..", "external", "LncPTPred", "dataset", "Final_lncRNA_Protein_Interaction.txt"
)
SEED = 42
KMERS = {k: ["".join(c) for c in itertools.product("ACGT", repeat=k)] for k in (1, 2, 3, 4)}
ALL_KMERS = [m for k in (1, 2, 3, 4) for m in KMERS[k]]


def featurize(seqs):
    """k-mer 1-4 counts, overlapping (the standard reading of 'k-mer counts')."""
    X = np.zeros((len(seqs), len(ALL_KMERS)), dtype=np.float32)
    idx = {m: i for i, m in enumerate(ALL_KMERS)}
    for r, s in enumerate(seqs):
        n = len(s)
        for k in (1, 2, 3, 4):
            for i in range(n - k + 1):
                j = idx.get(s[i:i + k])
                if j is not None:
                    X[r, j] += 1
    return X


def gc_frac(seqs):
    return np.array([(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs], dtype=np.float32)


STRUCT_NAMES = [
    "mfe", "mfe_per_nt", "ensemble_energy", "freq_mfe", "ensemble_diversity",
    "paired_frac", "n_helices", "mean_helix_len", "n_hairpins", "max_unpaired_run",
    "mfe_minus_ensemble",
]


def structure_features(seqs, cache=None):
    """Reconstruct LncPTPred's 'RNAfold secondary structure metrics' feature block.

    The paper says the lncRNA side has 351 features = k-mer 1-4 counts (340) plus RNAfold
    secondary-structure metrics, but does not enumerate the structure metrics. These 11 are a
    good-faith reconstruction of the standard RNAfold outputs; they are OUR choice, not the
    authors' list, and are reported as such.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if z["n"] == len(seqs):
            return z["X"]
    import RNA

    X = np.zeros((len(seqs), len(STRUCT_NAMES)), dtype=np.float32)
    for i, s in enumerate(seqs):
        fc = RNA.fold_compound(s)
        st, mfe = fc.mfe()
        _, ens = fc.pf()
        n = max(len(s), 1)
        paired = st.count("(") + st.count(")")
        helices, cur, lens = 0, 0, []
        for ch in st:
            if ch == "(":
                cur += 1
            elif cur:
                helices += 1
                lens.append(cur)
                cur = 0
        if cur:
            helices += 1
            lens.append(cur)
        runs, best, c = [], 0, 0
        for ch in st:
            if ch == ".":
                c += 1
                best = max(best, c)
            else:
                if c:
                    runs.append(c)
                c = 0
        X[i] = [
            mfe, mfe / n, ens, fc.pr_structure(st), fc.mean_bp_distance(),
            paired / n, helices, float(np.mean(lens)) if lens else 0.0,
            st.count("()") + sum(1 for j in range(len(st) - 1) if st[j] == "(" and st[j + 1] == "."),
            best, mfe - ens,
        ]
        if cache and (i + 1) % 50000 == 0:
            print(f"    structure features {i+1:,}/{len(seqs):,}", flush=True)
    if cache:
        np.savez_compressed(cache, X=X, n=len(seqs))
    return X


def cv_auroc(X, y, cat_feature=None, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = lgb.LGBMClassifier(random_state=seed, n_estimators=200, verbose=-1)
        fit_kw = {}
        if cat_feature is not None:
            fit_kw["categorical_feature"] = cat_feature
        clf.fit(X[tr], y[tr], **fit_kw)
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def gc_match(df, n_bins=20, seed=SEED):
    """Subsample so positives and negatives share a GC distribution (decile-style binning),
    mirroring the length/GC-matched control already used for RNAlight (exp_confound_matched.py)."""
    rng = np.random.default_rng(seed)
    gc = gc_frac(df["Sequence"].to_numpy())
    edges = np.quantile(gc, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    b = np.digitize(gc, edges[1:-1])
    y = df["Target"].to_numpy()
    keep = []
    for cell in np.unique(b):
        m = b == cell
        pos = np.where(m & (y == 1))[0]
        neg = np.where(m & (y == 0))[0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return df.iloc[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--sample", type=int, default=400_000,
                    help="stratified subsample size (memory bound; 0 = all rows)")
    ap.add_argument("--structure", action="store_true",
                    help="add reconstructed RNAfold secondary-structure metrics (needs ViennaRNA)")
    ap.add_argument("--out", default=os.path.join(HERE, "lncptpred_model_results.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.data, sep="\t")
    df["Sequence"] = df["Sequence"].str.strip().str.upper()
    n_full = len(df)
    if args.sample and args.sample < n_full:
        df = (
            df.groupby("Target")
            .sample(n=args.sample // 2, random_state=SEED)
            .reset_index(drop=True)
        )
        print(f"[stratified subsample: {len(df):,} rows of the full {n_full:,}]")

    results = []

    def run(name, X, y, cat=None):
        m, s = cv_auroc(X, y, cat_feature=cat)
        print(f"  {name:<22s} AUROC {m:.4f} ± {s:.4f}   ({X.shape[1]} features)", flush=True)
        results.append(dict(condition=name, auroc_mean=m, auroc_std=s,
                            n_features=int(X.shape[1]), n_rows=int(X.shape[0])))
        return m

    for tag, d in (("unmatched (as shipped)", df), ("GC-matched", gc_match(df))):
        print(f"\n=== {tag}: n={len(d):,}, positive rate {d['Target'].mean():.4f} ===", flush=True)
        seqs = d["Sequence"].to_numpy()
        y = d["Target"].to_numpy()
        prot = d["Protein"].astype("category").cat.codes.to_numpy().astype(np.float32)
        gc = gc_frac(seqs)
        length = np.array([len(s) for s in seqs], dtype=np.float32)

        run(f"gc-only [{tag}]", gc.reshape(-1, 1), y)
        run(f"len-gc-only [{tag}]", np.c_[length, gc], y)
        K = featurize(seqs)
        run(f"kmer-only [{tag}]", K, y)
        run(f"kmer+protein [{tag}]", np.c_[K, prot], y, cat=[K.shape[1]])
        if args.structure:
            S = structure_features(seqs, cache=None)
            run(f"structure-only [{tag}]", S, y)
            F = np.c_[K, S, prot]
            run(f"full (kmer+structure+protein) [{tag}]", F, y, cat=[F.shape[1] - 1])
            del S, F
        del K

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
