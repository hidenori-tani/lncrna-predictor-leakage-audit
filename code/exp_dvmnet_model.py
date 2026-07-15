#!/usr/bin/env python
"""Target B / DVMnet — run the DEEP MODEL ITSELF under alternative evaluation protocols.

This closes the gap left by exp_dvmnet_topology.py / exp_dvmnet_negatives.py, which showed the
*protocol* leaks node degree using zero-parameter heuristics but never ran DVMnet itself.

Conditions
  verbatim      : author's split (random-edge KFold(5, shuffle, seed=42)) + author's negatives
                  (torch_geometric.utils.negative_sampling over ALL node pairs)  -> reproduction gate
  random/uniform: author's split + uniform negatives restricted to valid lncRNA-miRNA non-edges
  random/degmat : author's split + configuration-model hard negatives (see make_neg_sampler)
  cold/uniform  : entity-disjoint cold-start split on lncRNA nodes + uniform negatives over held-out lncRNAs
  cold/degmat   : cold-start split + configuration-model hard negatives over held-out lncRNAs

For every condition we record BOTH
  best  : max validation AUROC over epochs (what upstream's `best_val_auc` tracking implies)
  final : last-epoch validation AUROC (no model selection on the evaluation fold)
The gap between them quantifies a third leakage axis: epoch selection on the evaluation split.

The DVMnet architecture, hyperparameters and features are imported verbatim from external/DVMnet.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import negative_sampling

import dvmnet_harness as H

SEED = 42
EPOCHS = 19          # upstream: `for epoch in range(1, 20)`
LR = 0.0008          # upstream
HIDDEN, OUT = 128, 64  # upstream: DVMNet(num_features, 128, 64)
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ negatives
def _global_degrees(edges):
    """Degree indexed by node id over all 804 nodes (upstream's negatives are not
    restricted to lncRNA-miRNA pairs, so both endpoints may be of either type)."""
    deg = np.zeros(H.N_NODES)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def make_neg_sampler(scheme, edge_set, deg, allowed_lnc):
    """Sample k negative (lnc, mi) pairs that are valid non-edges.

    scheme='uniform'  : lncRNA and miRNA drawn uniformly
    scheme='degmat'   : a CONFIGURATION-MODEL hard negative control. Endpoints are drawn
                        INDEPENDENTLY with probability proportional to GLOBAL degree, so the
                        negatives match the positives' *marginal* endpoint-degree distribution --
                        NOT the joint distribution of positive pairs. Three consequences worth
                        stating plainly, because each makes this control harder than the phrase
                        "degree-matched negatives" suggests:
                          (i)  it is a null, not a reconstruction of what the authors should have
                               sampled;
                          (ii) it is TRANSDUCTIVE -- `deg` comes from the full edge list, so the
                               sampler sees validation-fold positives' degrees. Deliberate: a null
                               built from training degree alone would leak the split back into the
                               negatives. But it is not a prospective drop-in for practitioners;
                          (iii) in a positive-unlabelled setting it preferentially draws (high-deg,
                               high-deg) non-edges, which are the pairs most likely to be
                               UNOBSERVED TRUE INTERACTIONS -- so results under it are a floor.
    allowed_lnc       : restrict lncRNA endpoints (cold-start uses only held-out lncRNAs)

    Rejection is against the FULL observed edge set, so a held-out positive is never sampled as a
    negative.
    """
    lnc_arr = np.asarray(sorted(allowed_lnc))
    mi_arr = np.arange(H.N_LNC, H.N_NODES)
    if scheme == "degmat":
        pl = deg[lnc_arr].astype(float)
        pm = deg[mi_arr].astype(float)
        # a zero-degree endpoint must still be reachable, else the sampler could never
        # place a negative on it; the epsilon keeps the profile but avoids p=0
        pl = pl + 1e-9 if pl.sum() > 0 else np.ones_like(pl)
        pm = pm + 1e-9 if pm.sum() > 0 else np.ones_like(pm)
        pl = pl / pl.sum()
        pm = pm / pm.sum()
    else:
        pl = pm = None

    def sample(k, seed):
        r = np.random.default_rng(seed)
        negs = set()
        guard = 0
        while len(negs) < k and guard < k * 10000:
            guard += 1
            u = int(r.choice(lnc_arr, p=pl))
            v = int(r.choice(mi_arr, p=pm))
            if (u, v) not in edge_set:
                negs.add((u, v))
        return np.array(sorted(negs))

    return sample


# ----------------------------------------------------------------- data/splits
def undirected(u, v):
    ui = [x for pair in zip(u, v) for x in pair]
    vi = [x for pair in zip(v, u) for x in pair]
    return torch.stack([torch.tensor(ui), torch.tensor(vi)], dim=0)


def folds_random_edge(edges):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in kf.split(edges):
        yield edges[tr], edges[va], set(range(H.N_LNC))


def folds_cold_start(edges):
    """Entity-disjoint on lncRNA nodes: every held-out lncRNA has zero degree in the train graph."""
    lnc_ids = np.arange(H.N_LNC)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_l, va_l in kf.split(lnc_ids):
        held = set(lnc_ids[va_l].tolist())
        mask = np.array([u in held for u, _ in edges])
        yield edges[~mask], edges[mask], held


# ---------------------------------------------------------------------- runner
def run_condition(feat, split, neg_scheme, verbose=False):
    edges = np.stack(H.edges_as_arrays(feat["node_table"]), axis=1)
    edges = np.stack(H.normalize_edge_orientation(edges[:, 0], edges[:, 1]), axis=1)
    edge_set = {(int(a), int(b)) for a, b in edges}
    deg = _global_degrees(edges)

    from model import DVMNet

    fold_iter = folds_random_edge(edges) if split == "random" else folds_cold_start(edges)
    best_aucs, final_aucs, base_aucs, impossible_frac = [], [], [], []

    for k, (tr_e, va_e, allowed_lnc) in enumerate(fold_iter):
        if len(va_e) == 0:
            continue
        H.seed_all(SEED + k)
        u_tr, v_tr = tr_e[:, 0], tr_e[:, 1]
        u_va, v_va = va_e[:, 0], va_e[:, 1]
        edge_index = undirected(u_tr.tolist(), v_tr.tolist())

        # ---- validation negatives (fixed before training, as upstream does)
        if neg_scheme == "verbatim":
            neg = negative_sampling(
                edge_index=edge_index, num_nodes=H.N_NODES, num_neg_samples=len(u_va),
                method="sparse",
            )
            neg_va = neg.t().numpy()
        else:
            sampler = make_neg_sampler(neg_scheme, edge_set, deg, allowed_lnc)
            neg_va = sampler(len(u_va), SEED + k)

        # upstream draws negatives over ALL node pairs, so some are lncRNA-lncRNA or
        # miRNA-miRNA -- pair types that cannot exist in this bipartite task. Quantify.
        # NB: (miRNA, lncRNA) is a VALID pair merely written in reverse orientation -- DVMnet's
        # score is a dot product (model.py:70-71) and hence symmetric -- so it is not counted here.
        bad = sum(1 for a, b in neg_va
                  if (a < H.N_LNC and b < H.N_LNC) or (a >= H.N_LNC and b >= H.N_LNC))
        impossible_frac.append(bad / len(neg_va))

        val_pos = torch.stack([torch.tensor(u_va), torch.tensor(v_va)], dim=0)
        val_neg = torch.tensor(neg_va.T)
        edge_label_index = torch.cat([val_pos, val_neg], dim=-1)
        edge_label = torch.cat([torch.ones(len(u_va)), torch.zeros(neg_va.shape[0])])

        # ---- zero-parameter degree baseline on the exact same fold + negatives
        tdeg = np.zeros(H.N_NODES)
        for a, b in tr_e:
            tdeg[a] += 1
            tdeg[b] += 1
        s = np.r_[
            [tdeg[a] + tdeg[b] for a, b in zip(u_va, v_va)],
            [tdeg[a] + tdeg[b] for a, b in neg_va],
        ]
        base_aucs.append(roc_auc_score(edge_label.numpy(), s))

        # ---- build Data objects exactly as upstream main.py does
        train_data = Data(
            x=feat["x_embedding"], x_lnc=feat["x_lnc"], x_mi=feat["x_mi"],
            edge_index=edge_index, edge_label=torch.ones(len(u_tr)),
            edge_label_index=torch.stack([torch.tensor(u_tr), torch.tensor(v_tr)], dim=0),
            xname=feat["graph_label"], di=feat["ser_di"].values, sub=feat["mi_sub"].values,
        )
        val_data = Data(
            x=feat["x_embedding"], edge_index=edge_index,
            edge_label=edge_label, edge_label_index=edge_label_index,
            xname=feat["graph_label"], di=feat["ser_di"].values, sub=feat["mi_sub"].values,
        )
        tf = NormalizeFeatures()
        train_data, val_data = tf(train_data), tf(val_data)

        model = DVMNet(train_data.num_features, HIDDEN, OUT)
        opt = torch.optim.Adam(params=model.parameters(), lr=LR)
        crit = torch.nn.BCEWithLogitsLoss()

        train_sampler = (
            None if neg_scheme == "verbatim"
            else make_neg_sampler(neg_scheme, edge_set, deg, set(range(H.N_LNC)))
        )

        aucs = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            opt.zero_grad()
            if train_sampler is None:
                nidx = negative_sampling(
                    edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                    num_neg_samples=train_data.edge_label_index.size(1), method="sparse",
                )
            else:
                nidx = torch.tensor(train_sampler(len(u_tr), SEED + 1000 * k + epoch).T)
            eli = torch.cat([train_data.edge_label_index, nidx], dim=-1)
            el = torch.cat([train_data.edge_label, train_data.edge_label.new_zeros(nidx.size(1))])
            out, *_ = model.mainNet(
                train_data.x, train_data.x_lnc, train_data.x_mi, train_data.edge_index,
                train_data.xname, eli, train_data.di, train_data.sub,
            )
            loss = crit(out.view(-1), el)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                vout, *_ = model.mainNet(
                    val_data.x, train_data.x_lnc, train_data.x_mi, val_data.edge_index,
                    val_data.xname, val_data.edge_label_index, val_data.di, val_data.sub,
                )
            auc = roc_auc_score(val_data.edge_label.numpy(), vout.view(-1).numpy())
            aucs.append(auc)
            if verbose:
                print(f"    fold{k} ep{epoch:02d} loss {loss.item():.4f} valAUC {auc:.4f}", flush=True)

        best_aucs.append(max(aucs))
        final_aucs.append(aucs[-1])
        print(
            f"  [{split}/{neg_scheme}] fold {k}: best {max(aucs):.4f} | final {aucs[-1]:.4f} "
            f"| degree-baseline {base_aucs[-1]:.4f} | n_val_pos {len(u_va)} "
            f"| impossible-type negatives {impossible_frac[-1]:.1%}",
            flush=True,
        )

    return dict(
        split=split, negatives=neg_scheme,
        best_mean=float(np.mean(best_aucs)), best_std=float(np.std(best_aucs)),
        final_mean=float(np.mean(final_aucs)), final_std=float(np.std(final_aucs)),
        degree_baseline_mean=float(np.mean(base_aucs)), degree_baseline_std=float(np.std(base_aucs)),
        impossible_pair_type_frac=float(np.mean(impossible_frac)),
        n_folds=len(best_aucs),
        # per-fold values are the substrate of every paired test we report; keep them in the
        # machine-readable artefact so no number in the paper depends on parsing the log
        best_per_fold=[float(v) for v in best_aucs],
        final_per_fold=[float(v) for v in final_aucs],
        degree_per_fold=[float(v) for v in base_aucs],
        impossible_frac_per_fold=[float(v) for v in impossible_frac],
    )


def main():
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="verbatim,random-uniform,random-degmat,cold-uniform,cold-degmat")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=os.path.join(HERE, "dvmnet_model_results.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    EPOCHS = args.epochs

    H.seed_all(SEED)
    feat = H.build_features(embed_cache=os.path.join(HERE, "dvmnet_embed_seed42.npz"))

    specs = {
        "verbatim": ("random", "verbatim"),
        "random-uniform": ("random", "uniform"),
        "random-degmat": ("random", "degmat"),
        "cold-uniform": ("cold", "uniform"),
        "cold-degmat": ("cold", "degmat"),
    }
    results = []
    for name in args.conditions.split(","):
        split, neg = specs[name]
        print(f"\n=== {name} (split={split}, negatives={neg}) ===", flush=True)
        t0 = time.time()
        r = run_condition(feat, split, neg, verbose=args.verbose)
        r["condition"] = name
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(
            f"  -> best {r['best_mean']:.4f} ± {r['best_std']:.4f} | "
            f"final {r['final_mean']:.4f} ± {r['final_std']:.4f} | "
            f"degree {r['degree_baseline_mean']:.4f} | impossible-neg {r['impossible_pair_type_frac']:.1%}"
            f" | {r['seconds']}s",
            flush=True,
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
