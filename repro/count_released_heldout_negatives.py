"""How often DVMnet's released training sampler presents a held-out positive as a negative (Section 2.4).

The released protocol draws its training negatives with negative_sampling over all node pairs, avoiding
only the edges of the training graph; the valid-pair scheme also rejects held-out positives. Removing the
impossible pair types (Q4) therefore changes this too. This replays the released draws -- the same call,
training graph, sample size and number of epochs, over the five partitions and five folds of the runs --
without training, and counts the draws that coincide, in either orientation, with a held-out interaction of
the fold. The released draws share their random stream with the model, so this replays the sampler, not the
exact draws of a run: the count estimates how often the overlap occurs.

    cd repro && OMP_NUM_THREADS=1 nice -n 19 ./venv/bin/python count_released_heldout_negatives.py \
        --out ../revision_r2/results/released_heldout_negatives.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.utils import negative_sampling

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dvmnet_harness as H  # noqa: E402
import exp_dvmnet_revision as M  # noqa: E402

REPEATS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(1)
    feat = H.build_features(embed_cache=os.path.join(HERE, "dvmnet_embed_seed42.npz"))
    edges = np.stack(H.edges_as_arrays(feat["node_table"]), axis=1)
    edges = np.stack(H.normalize_edge_orientation(edges[:, 0], edges[:, 1]), axis=1)

    draws = hits = impossible = 0
    per_fold = []
    for rep in range(REPEATS):
        cv_seed = M.SEED + 100 * rep
        for k, (tr_e, va_e, _, _) in enumerate(M.folds_random_edge(edges, cv_seed)):
            torch.manual_seed(cv_seed + k)
            held = {(int(a), int(b)) for a, b in va_e}
            edge_index = M.undirected(tr_e[:, 0].tolist(), tr_e[:, 1].tolist())
            fold_hits = 0
            for _ in range(M.EPOCHS):
                nidx = negative_sampling(edge_index=edge_index, num_nodes=H.N_NODES,
                                         num_neg_samples=len(tr_e), method="sparse").numpy()
                for a, b in zip(nidx[0], nidx[1]):
                    a, b = int(a), int(b)
                    if (a < H.N_LNC) == (b < H.N_LNC):
                        impossible += 1
                    elif (min(a, b), max(a, b)) in held:
                        fold_hits += 1
                draws += nidx.shape[1]
            hits += fold_hits
            per_fold.append(fold_hits)
    out = dict(partitions=REPEATS, folds=len(per_fold), epochs=M.EPOCHS, draws=draws,
               heldout_hits=hits, heldout_hit_fraction=hits / draws,
               heldout_hits_per_fold_mean=float(np.mean(per_fold)), heldout_hits_per_fold=per_fold,
               impossible_fraction=impossible / draws, seconds=round(time.time() - t0, 1))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print({k: v for k, v in out.items() if k != "heldout_hits_per_fold"}, flush=True)


if __name__ == "__main__":
    main()
