#!/usr/bin/env python
"""Target C / LncPTPred (Brief Bioinform 2025, bbaf432) — dataset-level audit.

LncPTPred predicts lncRNA-protein interaction from (protein, RNA fragment) pairs, where
fragments come from a shifting window over lncRNA loci and "real negatives" are CLIP-Seq
non-bound fragments. The shipped benchmark
(dataset/Final_lncRNA_Protein_Interaction.txt) has columns: Protein, Sequence, Strand, Target.

Probes run here (all model-free, so they characterise the BENCHMARK, not one implementation):
  P0  duplicate audit      : exact-duplicate fragments, and fragments carrying BOTH labels
  P1  trivial baselines    : protein identity alone / length / GC / strand -- no RNA sequence read
  P2  entity leakage       : the file carries no transcript ID, so a transcript-disjoint split is
                             not even expressible; quantify near-duplicate structure instead
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(
    HERE, "..", "external", "LncPTPred", "dataset", "Final_lncRNA_Protein_Interaction.txt"
)


def load(path):
    df = pd.read_csv(path, sep="\t", dtype={"Protein": "category", "Sequence": str,
                                            "Strand": "category", "Target": np.int8})
    df["Sequence"] = df["Sequence"].str.strip().str.upper()
    return df


def p0_duplicates(df):
    print("\n=== P0: duplicate audit ===")
    n = len(df)
    uniq_seq = df["Sequence"].nunique()
    print(f"rows                         : {n:,}")
    print(f"unique fragment sequences    : {uniq_seq:,}  ({uniq_seq/n:.1%} of rows)")
    print(f"exact-duplicate rows (seq)   : {n - uniq_seq:,}  ({1 - uniq_seq/n:.1%})")

    # same (protein, sequence) pair appearing more than once
    pair_counts = df.groupby(["Protein", "Sequence"], observed=True).size()
    dup_pairs = int((pair_counts > 1).sum())
    print(f"(protein,seq) pairs repeated : {dup_pairs:,} of {len(pair_counts):,} unique pairs")

    # same (protein, sequence) carrying BOTH labels -> irreducible label contradiction
    lab = df.groupby(["Protein", "Sequence"], observed=True)["Target"].nunique()
    contradictory = int((lab > 1).sum())
    print(f"(protein,seq) with BOTH 0 and 1 labels: {contradictory:,}"
          f"  ({contradictory/len(lab):.3%} of unique pairs)")

    # a sequence used with many proteins
    per_seq_prot = df.groupby("Sequence", observed=True)["Protein"].nunique()
    print(f"fragments reused across proteins: {int((per_seq_prot>1).sum()):,} "
          f"(max {int(per_seq_prot.max())} proteins)")
    return dict(rows=n, unique_seq=uniq_seq, dup_pairs=dup_pairs, contradictory=contradictory)


def p1_trivial(df, seed=42):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    print("\n=== P1: trivial baselines (no RNA sequence content read) ===")
    y = df["Target"].to_numpy()
    print(f"class balance: positives {y.mean():.4f}  (n_pos {int(y.sum()):,} / n {len(y):,})")

    prot = df["Protein"].to_numpy()
    seqs = df["Sequence"].to_numpy()
    length = np.array([len(s) for s in seqs], dtype=float)
    gc = np.array([(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs])
    strand = (df["Strand"].to_numpy() == "+").astype(float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = defaultdict(list)
    for tr, te in skf.split(np.zeros(len(y)), y):
        # protein-identity-only: score = positive rate of that protein in TRAIN
        rate = defaultdict(lambda: y[tr].mean())
        tmp = defaultdict(list)
        for p, t in zip(prot[tr], y[tr]):
            tmp[p].append(t)
        for p, v in tmp.items():
            rate[p] = float(np.mean(v))
        s_prot = np.array([rate[p] for p in prot[te]])
        scores["protein identity only"].append(roc_auc_score(y[te], s_prot))
        scores["fragment length only"].append(roc_auc_score(y[te], length[te]))
        scores["GC content only"].append(roc_auc_score(y[te], gc[te]))
        scores["strand only"].append(roc_auc_score(y[te], strand[te]))

    for k, v in scores.items():
        v = np.array(v)
        print(f"  {k:<24s} AUROC {v.mean():.4f} ± {v.std():.4f}")
    return {k: float(np.mean(v)) for k, v in scores.items()}


def p2_length_by_class(df):
    print("\n=== P2: is the label confounded with trivial fragment properties? ===")
    from scipy.stats import mannwhitneyu

    seqs = df["Sequence"].to_numpy()
    y = df["Target"].to_numpy()
    length = np.array([len(s) for s in seqs], dtype=float)
    gc = np.array([(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs])
    for name, arr in [("length (nt)", length), ("GC fraction", gc)]:
        a, b = arr[y == 1], arr[y == 0]
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {name:<14s} positive median {np.median(a):.4f} | negative median "
              f"{np.median(b):.4f} | MWU p={p:.3g}")
    print(f"  distinct fragment lengths: {len(set(length.astype(int)))} "
          f"(min {int(length.min())}, max {int(length.max())})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--sample", type=int, default=0, help="subsample N rows (0 = all)")
    args = ap.parse_args()

    df = load(args.data)
    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"[subsampled to {len(df):,} rows]")

    p0_duplicates(df)
    p1_trivial(df)
    p2_length_by_class(df)


if __name__ == "__main__":
    main()
