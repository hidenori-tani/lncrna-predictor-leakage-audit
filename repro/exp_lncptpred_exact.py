#!/usr/bin/env python
"""Target C / LncPTPred -- rebuild the authors' feature construction EXACTLY, from their released
feature-generation code, and re-run the model-level comparisons on it.

Reviewer 2, point 4: "The reconstructed model differs from the original LncPTPred and achieves an
AUROC of approximately 0.79 compared with the reported > 0.9. Further validation is needed to
establish whether the subsequent analyses and conclusions are representative of the original
model."

The submitted version reconstructed the feature family from the *paper's prose*, because the paper
does not enumerate it.  The released repository does: `src/gen_lncrna_feat.py` and
`src/gen_protein_feat.py` define both feature sets completely, and their sizes reproduce the two
counts the paper states.

  lncRNA side (351) = 340 k-mer FRACTIONS (k = 1..4 over ACGT)
                    +   9 secondary-structure metrics from the RNAfold dot-bracket
                        (total_base_pairs, unpaired_bases, stem_count, avg_stem_length,
                         hairpin_count, avg_hairpin_size, bulge_count, multiloop_count,
                         base_pair_density)
                    +   1 mfe
                    +   1 Strand                                   = 351
  protein side (369) =  20 amino-acid fractions
                     +  39 hydropathy    (3 classes: 3 + 3^2 + 3^3)
                     + 155 chemical      (5 classes: 5 + 5^2 + 5^3)
                     + 155 volume        (5 classes: 5 + 5^2 + 5^3)  = 369

Three differences between the submitted reconstruction and the released construction are therefore
now correctable, and each is measured separately below:
  (1) we used k-mer COUNTS, the authors use FRACTIONS (counts carry fragment length, fractions do
      not, and these fragments vary from 10 to 198 nt);
  (2) our 11 structure metrics were our own choice of standard RNAfold outputs, not theirs;
  (3) we did not use Strand as a feature; they do.

Everything else -- the stratified subsample and its seed, the 5-fold stratified CV, and the
LightGBM settings -- is identical to exp_lncptpred_model.py, so the new numbers are directly
comparable with the submitted ones.
"""
import argparse
import itertools
import json
import os
import re
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_JOBS = 4   # half the logical cores: the machine is shared with the author's own work
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "external", "LncPTPred", "dataset")

KMERS = {k: ["".join(p) for p in itertools.product("ACGT", repeat=k)] for k in range(1, 5)}
ALL_KMERS = [m for k in range(1, 5) for m in KMERS[k]]          # 4 + 16 + 64 + 256 = 340

STRUCT_NAMES = ["total_base_pairs", "unpaired_bases", "stem_count", "avg_stem_length",
                "hairpin_count", "avg_hairpin_size", "bulge_count", "multiloop_count",
                "base_pair_density", "mfe"]


# ------------------------------------------------- lncRNA side (released gen_lncrna_feat.py)
def kmer_fractions(seqs):
    """Their get_kmer_fractions: counts / (len(seq) - k + 1), zero where the window does not fit."""
    idx = {m: i for i, m in enumerate(ALL_KMERS)}
    X = np.zeros((len(seqs), len(ALL_KMERS)), dtype=np.float32)
    for r, s in enumerate(seqs):
        n = len(s)
        for k in (1, 2, 3, 4):
            total = n - k + 1
            if total <= 0:
                continue
            for i in range(total):
                j = idx.get(s[i:i + k])
                if j is not None:
                    X[r, j] += 1.0
            for m in KMERS[k]:
                X[r, idx[m]] /= total
    return X


def extract_structure_features(dot_bracket):
    """Verbatim port of extract_structure_features() in src/gen_lncrna_feat.py."""
    total_base_pairs = dot_bracket.count("(")
    unpaired_bases = dot_bracket.count(".")
    stem_matches = re.findall(r"\(+", dot_bracket)
    stem_count = len(stem_matches)
    avg_stem_length = sum(len(m) for m in stem_matches) / stem_count if stem_count else 0.0
    hairpins = re.findall(r"\([^()\.]{0,1}\.*[^()\.]{0,1}\)", dot_bracket)
    hairpin_count = len(hairpins)
    avg_hairpin_size = (sum(m.count(".") for m in hairpins) / hairpin_count) if hairpin_count else 0.0
    bulges = re.findall(r"\(\.+\(", dot_bracket) + re.findall(r"\)\.+\)", dot_bracket)
    bulge_count = len(bulges)
    multiloop_count = max(len(re.findall(r"\)[.]+\(", dot_bracket)) - 1, 0)
    base_pair_density = round(dot_bracket.count("(") / len(dot_bracket), 4)
    return [total_base_pairs, unpaired_bases, stem_count, round(avg_stem_length, 2),
            hairpin_count, round(avg_hairpin_size, 2), bulge_count, multiloop_count,
            base_pair_density]


def structure_block(seqs, cache=None):
    """9 structure metrics + mfe.  The released code shells out to `RNAfold --noPS` and parses the
    printed energy, which is fixed to two decimals; we call the ViennaRNA 2.7.2 Python API and
    round identically so the two paths agree."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if int(z["n"]) == len(seqs):
            print(f"    [structure cache hit: {cache}]", flush=True)
            return z["X"]
    import RNA
    X = np.zeros((len(seqs), len(STRUCT_NAMES)), dtype=np.float32)
    t0 = time.time()
    for i, s in enumerate(seqs):
        db, mfe = RNA.fold(s)
        X[i, :9] = extract_structure_features(db)
        X[i, 9] = round(mfe, 2)
        if (i + 1) % 100_000 == 0:
            print(f"    structure {i+1:,}/{len(seqs):,}  ({time.time()-t0:.0f}s)", flush=True)
    if cache:
        np.savez_compressed(cache, X=X, n=len(seqs))
    return X


# ------------------------------------------------ protein side (released gen_protein_feat.py)
AA = list("ACDEFGHIKLMNPQRSTVWY")
GROUPINGS = {
    "hydropathy": {"1": list("ACILMFWV"), "2": list("GHPSTY"), "3": list("RNDQEK")},
    "chemical":   {"1": list("GAVLIP"), "2": list("FYW"), "3": list("STNQ"),
                   "4": list("CM"), "5": list("DEHKR")},
    "volume":     {"1": list("AGS"), "2": list("NDCPT"), "3": list("QEHV"),
                   "4": list("RILKM"), "5": list("FWY")},
}


def protein_features():
    """Return (DataFrame indexed by protein name, list of 369 column names)."""
    df = pd.read_csv(os.path.join(DATA, "Final_Protein.txt"), sep="\t")
    df["Protein_Sequence"] = df["Protein_Sequence"].str.upper()
    out = pd.DataFrame(index=df["Protein"].values)

    for aa in AA:                                                   # 20
        out[f"{aa}_fraction"] = [s.count(aa) / len(s) for s in df["Protein_Sequence"]]

    for props, groups in GROUPINGS.items():                         # 39 + 155 + 155
        table = str.maketrans({a: g for g, members in groups.items() for a in members})
        red = [s.translate(table) for s in df["Protein_Sequence"]]
        classes = sorted(groups)
        for n in (1, 2, 3):
            for combo in itertools.product(classes, repeat=n):
                pat = "".join(combo)
                col = f"{pat}_{props}_prot_fraction"
                if n == 1:
                    out[col] = [s.count(pat) / len(s) for s in red]
                else:
                    out[col] = [len(re.findall("(?=" + pat + ")", s)) / (len(s) - (n - 1))
                                for s in red]
    return out, list(out.columns)


# ------------------------------------------------------------------------------- evaluation
def cv_auroc(X, y, cat_feature=None, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = lgb.LGBMClassifier(random_state=seed, n_estimators=200, verbose=-1,
                                 n_jobs=N_JOBS)
        fit_kw = {}
        if cat_feature is not None:
            fit_kw["categorical_feature"] = cat_feature
        clf.fit(X[tr], y[tr], **fit_kw)
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs)), [float(a) for a in aucs]


def gc_match(df, gc, n_bins=20, seed=SEED):
    """Same decile-style GC matching as exp_lncptpred_model.py, so the channel-closing comparison
    is the one the submitted version reports, only on exact features."""
    rng = np.random.default_rng(seed)
    bins = np.clip((gc * n_bins).astype(int), 0, n_bins - 1)
    keep = []
    y = df["Target"].to_numpy()
    for b in range(n_bins):
        ip = np.where((bins == b) & (y == 1))[0]
        ineg = np.where((bins == b) & (y == 0))[0]
        m = min(len(ip), len(ineg))
        if m == 0:
            continue
        keep.append(rng.choice(ip, m, replace=False))
        keep.append(rng.choice(ineg, m, replace=False))
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--out", default=os.path.join(HERE, "lncptpred_exact_results.json"))
    ap.add_argument("--struct-cache", default=os.path.join(HERE, "lncptpred_exact_struct.npz"))
    args = ap.parse_args()

    prot_df, prot_cols = protein_features()
    print(f"protein features: {len(prot_cols)} 列（論文の記載は 369）", flush=True)
    assert len(prot_cols) == 369, len(prot_cols)

    df = pd.read_csv(os.path.join(DATA, "Final_lncRNA_Protein_Interaction.txt"), sep="\t")
    n_full = len(df)
    if args.sample and args.sample < n_full:
        df = (df.groupby("Target", group_keys=False)
                .sample(n=args.sample // 2, random_state=SEED)
                .reset_index(drop=True))
    print(f"rows: {len(df):,} of {n_full:,}  positive rate {df.Target.mean():.4f}", flush=True)

    seqs = df["Sequence"].str.upper().tolist()
    y = df["Target"].to_numpy()

    t0 = time.time()
    K = kmer_fractions(seqs)
    print(f"k-mer fractions: {K.shape}  ({time.time()-t0:.0f}s)", flush=True)
    S = structure_block(seqs, cache=args.struct_cache)
    print(f"structure block: {S.shape}", flush=True)
    strand = df["Strand"].map({"+": 1, "-": -1}).to_numpy(dtype=np.float32).reshape(-1, 1)
    LNC = np.c_[K, S, strand].astype(np.float32)
    print(f"lncRNA side: {LNC.shape[1]} 列（論文の記載は 351）", flush=True)
    assert LNC.shape[1] == 351, LNC.shape

    P369 = prot_df.loc[df["Protein"].values, prot_cols].to_numpy(dtype=np.float32)
    protcat = df["Protein"].astype("category").cat.codes.to_numpy().astype(np.float32).reshape(-1, 1)

    gc = np.array([(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs], dtype=np.float32)
    keep = gc_match(df, gc)
    print(f"GC-matched subset: {len(keep):,} rows", flush=True)

    results = []

    def run(name, X, yy, cat=None):
        t = time.time()
        m, s, folds = cv_auroc(X, yy, cat_feature=cat)
        results.append(dict(condition=name, auroc_mean=m, auroc_std=s, per_fold=folds,
                            n=int(len(yy)), n_features=int(X.shape[1]),
                            seconds=round(time.time() - t, 1)))
        print(f"  {name:<44} {m:.4f} ± {s:.4f}   (n={len(yy):,}, {X.shape[1]} feat, "
              f"{results[-1]['seconds']}s)", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== released feature construction (exact) ===", flush=True)
    run("exact lncRNA 351 only", LNC, y)
    run("exact lncRNA 351 + protein 369", np.c_[LNC, P369], y)
    run("exact lncRNA 351 + protein categorical", np.c_[LNC, protcat], y,
        cat=[LNC.shape[1]])

    print("\n=== isolating the three differences from the submitted reconstruction ===", flush=True)
    run("released minus structure/strand (k-mer fractions + protein 369)",
        np.c_[K, P369], y)
    run("released minus strand (351-1 + protein 369)",
        np.c_[K, S, P369], y)

    print("\n=== composition channel, on exact features ===", flush=True)
    run("exact full [GC-matched]", np.c_[LNC, P369][keep], y[keep])

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
