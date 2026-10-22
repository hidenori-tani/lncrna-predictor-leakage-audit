#!/usr/bin/env python
"""Replay every 'swap' negative draw of the R2 DVMnet runs with the runs' own seeds and verify it.

For each partition (cv seed 42 + 100 r, r = 0..4), each fold k, and both splits, the draws are the
ones exp_dvmnet_revision.run_fold makes: training negatives at every epoch (seed + 1000 + epoch,
epochs 1..19, from the training positives), and the selection and test halves (seed + 11 and
seed + 12, from each half's positives), where seed = cv seed + k.

Checked for every draw: no negative is an observed interaction; no negative repeats; every lncRNA
appears among the negatives exactly as often as among the positives; every miRNA does too unless
the draw had to re-draw pairs; under the entity-disjoint split, training negatives touch no held-out
lncRNA and each evaluation half's negatives stay within that half's lncRNAs.

Re-drawn pairs (added after the Codex review of 2026-10-22, which pointed out that the previous form
accepted any re-draw): a training set or a test half must be exact, because the manuscript says so;
any other draw may re-draw only as many pairs as no exact re-pairing can avoid. That minimum is the
number of positives minus the maximum flow of the network source -> lncRNA (capacity = its number
of positives) -> miRNA (capacity 1 where the pair is not a recorded interaction) -> sink (capacity =
the miRNA's number of positives), and the draw fails if it re-draws more.

Output: results/swap_draws_check.json (counts of draws, pairs, and re-drawn pairs per split and set).
"""
import json
import os
import sys
import time
from collections import Counter

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.join(HERE, "..", "repro")
sys.path.insert(0, REPRO)
import dvmnet_harness as H            # noqa: E402
import exp_dvmnet_revision as R       # noqa: E402

REPEATS = 5


def min_redrawn(pos, edge_set):
    """Fewest pairs that any exact re-pairing of these positives must leave unmatched (max flow)."""
    G = nx.DiGraph()
    kl, cm = Counter(pos[:, 0].tolist()), Counter(pos[:, 1].tolist())
    for lnc, k in kl.items():
        G.add_edge("s", ("l", lnc), capacity=k)
    for mi, c in cm.items():
        G.add_edge(("m", mi), "t", capacity=c)
    for lnc in kl:
        for mi in cm:
            if (lnc, mi) not in edge_set:
                G.add_edge(("l", lnc), ("m", mi), capacity=1)
    return len(pos) - nx.maximum_flow_value(G, "s", "t")


def main():
    t0 = time.time()
    feat = H.build_features(embed_cache=os.path.join(REPRO, "dvmnet_embed_seed42.npz"))
    edges = np.stack(H.edges_as_arrays(feat["node_table"]), axis=1)
    edges = np.stack(H.normalize_edge_orientation(edges[:, 0], edges[:, 1]), axis=1)
    edge_set = {(int(a), int(b)) for a, b in edges}
    deg = R._global_degrees(edges)
    tally = {}
    for split in ("random", "cold"):
        for rep in range(REPEATS):
            cv = R.SEED + 100 * rep
            it = R.folds_random_edge(edges, cv) if split == "random" else R.folds_cold_start(edges, cv)
            for k, (tr_e, va_e, _allowed, held) in enumerate(it):
                seed = cv + k
                rng = np.random.default_rng(seed)
                m_sel, m_test, l_sel, l_test = R.partition_eval(split, va_e, held, rng)
                draws = [("train", tr_e, seed + 1000 + ep, None) for ep in range(1, R.EPOCHS + 1)]
                draws += [("selection", va_e[m_sel], seed + 11, l_sel), ("test", va_e[m_test], seed + 12, l_test)]
                for name, pos, s, allowed in draws:
                    sampler = R.make_neg_sampler("swap", edge_set, deg, set(), pos_pairs=pos)
                    neg = sampler(len(pos), s)
                    st = sampler.stats
                    pairs = [(int(a), int(b)) for a, b in neg]
                    assert not any(p in edge_set for p in pairs), "observed interaction among negatives"
                    assert len(set(pairs)) == len(pairs), "repeated negative"
                    assert Counter(neg[:, 0].tolist()) == Counter(pos[:, 0].tolist()), "lncRNA counts"
                    exact_mi = Counter(neg[:, 1].tolist()) == Counter(pos[:, 1].tolist())
                    assert exact_mi or st["relaxed"] > 0, "miRNA counts differ without a re-draw"
                    if name in ("train", "test"):
                        assert st["relaxed"] == 0 and exact_mi, f"{split}/{name} draw is not exact"
                    floor = min_redrawn(pos, edge_set) if st["relaxed"] else 0
                    assert st["relaxed"] == floor, f"re-drew {st['relaxed']} pairs where {floor} is the minimum"
                    if split == "cold":
                        if name == "train":
                            assert not set(neg[:, 0].tolist()) & {int(x) for x in held}
                        else:
                            assert set(neg[:, 0].tolist()) <= {int(x) for x in allowed}
                    a = tally.setdefault(f"{split}/{name}", dict(draws=0, pairs=0, redrawn=0, redrawn_minimum=0,
                                                                  outside_pool=0, draws_exact=0))
                    a["redrawn_minimum"] += floor
                    if st["relaxed"]:
                        top = max(Counter(pos[:, 0].tolist()).values()) / len(pos)
                        a.setdefault("redrawn_draws_top_lncrna_share", []).append(round(top, 4))
                    a["draws"] += 1
                    a["pairs"] += st["n"]
                    a["redrawn"] += st["relaxed"]
                    a["outside_pool"] += st["outside_pool"]
                    a["draws_exact"] += int(st["relaxed"] == 0)
    for a in tally.values():
        a["redrawn_pct"] = 100.0 * a["redrawn"] / a["pairs"]
    out = dict(repeats=REPEATS, epochs=R.EPOCHS, sets=tally, seconds=round(time.time() - t0, 1),
               checks=["no observed interaction", "no repeat", "lncRNA counts exact",
                       "miRNA counts exact unless re-drawn", "training sets and test halves exact",
                       "re-drawn pairs = the max-flow minimum", "cold: entities kept"])
    path = os.path.join(HERE, "results", "swap_draws_check.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    for key, a in tally.items():
        print(f"  {key:16s} draws {a['draws']:4d}  exact {a['draws_exact']:4d}  pairs {a['pairs']:6d}  "
              f"re-drawn {a['redrawn']:4d} ({a['redrawn_pct']:.2f}%)")
    print(f"all checks passed ({out['seconds']}s) -> {path}")


if __name__ == "__main__":
    main()
