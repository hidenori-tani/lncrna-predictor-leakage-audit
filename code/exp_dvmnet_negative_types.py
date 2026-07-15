#!/usr/bin/env python
"""Target B / DVMnet — what pair TYPES does the released negative sampler actually produce?

main.py draws negatives with torch_geometric.utils.negative_sampling(num_nodes=data.num_nodes),
i.e. uniformly over all 804x804 node pairs, not over the bipartite lncRNA x miRNA pair space.
Nodes 0..283 are lncRNA, 284..803 are miRNA.

DVMnet scores a pair as a dot product, `out = (x[i] * x[j]).sum(-1)` (model.py:70-71), which is
SYMMETRIC -- so a (miRNA, lncRNA) pair is a valid lncRNA-miRNA pair merely written in reverse
orientation, and must NOT be counted as an impossible type. Only lncRNA-lncRNA and miRNA-miRNA
pairs are impossible in this bipartite task. This script measures that distinction exactly.
"""
import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch_geometric.utils import negative_sampling

import dvmnet_harness as H

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))


def classify(pairs):
    a, b = pairs[:, 0], pairs[:, 1]
    a_lnc, b_lnc = a < H.N_LNC, b < H.N_LNC
    return dict(
        lnc_lnc=int(np.sum(a_lnc & b_lnc)),
        mi_mi=int(np.sum(~a_lnc & ~b_lnc)),
        lnc_mi=int(np.sum(a_lnc & ~b_lnc)),
        mi_lnc=int(np.sum(~a_lnc & b_lnc)),
    )


def main():
    H.seed_all(SEED)
    feat = H.build_features(embed_cache=os.path.join(HERE, "dvmnet_embed_seed42.npz"))
    edges = np.stack(H.edges_as_arrays(feat["node_table"]), axis=1)
    edges = np.stack(H.normalize_edge_orientation(edges[:, 0], edges[:, 1]), axis=1)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    tot = dict(lnc_lnc=0, mi_mi=0, lnc_mi=0, mi_lnc=0)
    for k, (tr, va) in enumerate(kf.split(edges)):
        H.seed_all(SEED + k)
        u, v = edges[tr][:, 0].tolist(), edges[tr][:, 1].tolist()
        ui = [x for pair in zip(u, v) for x in pair]
        vi = [x for pair in zip(v, u) for x in pair]
        edge_index = torch.stack([torch.tensor(ui), torch.tensor(vi)], dim=0)
        neg = negative_sampling(
            edge_index=edge_index, num_nodes=H.N_NODES,
            num_neg_samples=len(va), method="sparse",
        ).t().numpy()
        c = classify(neg)
        for key in tot:
            tot[key] += c[key]
        n = len(neg)
        imp = (c["lnc_lnc"] + c["mi_mi"]) / n
        print(f"fold {k}: n={n} | lnc-lnc {c['lnc_lnc']:4d} | mi-mi {c['mi_mi']:4d} | "
              f"lnc-mi {c['lnc_mi']:4d} | mi-lnc {c['mi_lnc']:4d} | IMPOSSIBLE type {imp:.1%}")

    n = sum(tot.values())
    impossible = tot["lnc_lnc"] + tot["mi_mi"]
    valid = tot["lnc_mi"] + tot["mi_lnc"]
    print("\n--- pooled over 5 folds ---")
    print(f"total negatives sampled     : {n}")
    print(f"  lncRNA-lncRNA (impossible): {tot['lnc_lnc']:5d}  {tot['lnc_lnc']/n:.1%}")
    print(f"  miRNA-miRNA   (impossible): {tot['mi_mi']:5d}  {tot['mi_mi']/n:.1%}")
    print(f"  IMPOSSIBLE pair types     : {impossible:5d}  {impossible/n:.1%}")
    print(f"  valid lncRNA-miRNA        : {valid:5d}  {valid/n:.1%}"
          f"   (of which {tot['mi_lnc']/n:.1%} written in reverse orientation;")
    print(f"                                the model's dot-product score is symmetric, so"
          f" orientation is irrelevant)")

    nl, nm, N = H.N_LNC, H.N_MI, H.N_NODES
    print("\n--- analytic expectation for uniform sampling over all ordered node pairs ---")
    print(f"  P(lnc-lnc) = ({nl}/{N})^2                = {(nl/N)**2:.3f}")
    print(f"  P(mi-mi)   = ({nm}/{N})^2                = {(nm/N)**2:.3f}")
    print(f"  P(impossible)                            = {(nl/N)**2 + (nm/N)**2:.3f}")
    print(f"  P(valid lnc-mi, either orientation)      = {2*(nl/N)*(nm/N):.3f}")


if __name__ == "__main__":
    main()
