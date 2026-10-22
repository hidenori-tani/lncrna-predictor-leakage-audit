#!/usr/bin/env python
"""Target B / DVMnet benchmark -- several model families on the SAME task, the SAME folds and the
SAME test negatives as the R1 protocol ablation.

Reviewer 2 (R1 round), point 1: "... it would be beneficial to compare several baseline models on
the same task and dataset." The same reviewer's objection is that degree-matched negatives and
entity-disjoint splits may change what is being predicted rather than correct a defect. Running
models that differ in what information they CAN use, through the identical protocol changes, is
what separates the two readings:

  degree_sum      zero parameters: deg_train(u) + deg_train(v)                     (R1's P1 baseline)
  popularity_lr   logistic regression on log-degrees of the two endpoints and their product
  node_id_lr      logistic regression on one-hot endpoint identities only: it can learn WHO is
                  frequently paired and nothing about any pair, so it is a positive control for
                  "a score attainable without interaction-specific information"
  composition_gbm LightGBM on k-mer (k = 1-3) fractions and lengths of both sequences; no graph
  doc2vec_gbm     LightGBM on DVMnet's own doc2vec node features of both endpoints; no graph
  seed_sites      zero parameters: canonical 7mer-m8 sites of the miRNA seed (nt 2-8) in the lncRNA,
                  a biologically motivated score that reads the pair, not the nodes
  seed_family_lr  (R2) logistic regression on the counts of the five canonical site types (8mer,
                  7mer-m8, 7mer-A1, 6mer, offset 6mer) and the lncRNA length: a learned model that
                  reads the pair. Under the R2 'swap' negatives every molecule is met equally often in
                  both classes, so the models above that score each molecule separately are at chance
                  by construction there, and the pair-reading models are the comparison that remains
  DVMnet          the nested estimate from the R1 run (repro/dvmnet_revision_results.json)

Test positives, test negatives and folds are regenerated with exactly the calls and seeds of
`exp_dvmnet_revision.run_fold`, and the regenerated degree-sum AUROC must equal the stored R1 value
in every fold (checked; the run aborts otherwise). Training negatives for the learned baselines are
drawn from the same scheme DVMnet trains on (its 19 per-epoch draws for the matched schemes; for the
released scheme, torch_geometric's sampler with fixed seeds, since DVMnet's own draws depend on
model state and cannot be replayed without the model).
"""
import argparse
import json
import os
import string

# torch and LightGBM each bring their own libomp on macOS; with both runtimes' worker pools alive
# in one process the run deadlocks in __kmp_hyper_barrier_release (observed 2026-10-22). The models
# here are small, so every library runs single-threaded and no OpenMP worker pool is ever started.
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dvmnet_harness as H
import exp_dvmnet_revision as R
from torch_geometric.utils import negative_sampling

CONDITIONS = [("random", "verbatim"), ("random", "uniform"), ("random", "degmat"),
              ("cold", "uniform"), ("cold", "degmat"),
              ("random", "swap"), ("cold", "swap")]          # R2: positives' own endpoints re-paired
REPEATS = 5
EPOCHS = R.EPOCHS
KMERS = {k: ["".join(p) for p in itertools.product("ACGT", repeat=k)] for k in (1, 2, 3)}


# ------------------------------------------------------------------------------ node descriptors
def node_sequences(feat):
    """name -> sequence, cleaned exactly as dvmnet_harness._read_sequences cleans them."""
    mt = pd.read_csv(os.path.join(H.DVMNET_DIR, "dataset", "mirna_lncrna_interaction.csv"))
    seq = {}
    for name, s in zip(mt["lncrna"], mt["lncrna_seq"]):
        if name not in seq:
            seq[name] = str(s).translate(str.maketrans("", "", string.punctuation)).upper().replace("U", "T")
    for name, s in zip(mt["mirna"], mt["mirna_seq"]):
        if name not in seq:
            s = str(s).replace(".", "")
            seq[name] = (s.split(",")[0] if "," in s else s).upper().replace("U", "T")
    out = [seq.get(n, "") for n in feat["graph_label"]]
    missing = sum(1 for s in out if not s)
    return out, missing


def kmer_frac(s):
    v = []
    for k in (1, 2, 3):
        tot = max(len(s) - k + 1, 1)
        c = {m: 0 for m in KMERS[k]}
        for i in range(len(s) - k + 1):
            w = s[i:i + k]
            if w in c:
                c[w] += 1
        v.extend(c[m] / tot for m in KMERS[k])
    return v


def revcomp(s):
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


def _count(seq, site):
    n, start = 0, 0
    while True:
        j = seq.find(site, start)
        if j < 0:
            return n
        n += 1
        start = j + 1


def seed_site_count(lnc_seq, mi_seq):
    """Canonical 7mer-m8 site: the reverse complement of miRNA nt 2-8, counted with overlaps."""
    if len(mi_seq) < 8 or not lnc_seq:
        return 0
    return _count(lnc_seq, revcomp(mi_seq[1:8]))


def seed_family_counts(lnc_seq, mi_seq):
    """Overlapping counts of the canonical site types in the lncRNA (target strand, 5'->3'):
    8mer (nt 2-8 + A1), 7mer-m8 (nt 2-8), 7mer-A1 (nt 2-7 + A1), 6mer (nt 2-7), offset 6mer (nt 3-8)."""
    if len(mi_seq) < 8 or not lnc_seq:
        return (0, 0, 0, 0, 0)
    m28, m27, m38 = revcomp(mi_seq[1:8]), revcomp(mi_seq[1:7]), revcomp(mi_seq[2:8])
    return (_count(lnc_seq, m28 + "A"), _count(lnc_seq, m28), _count(lnc_seq, m27 + "A"),
            _count(lnc_seq, m27), _count(lnc_seq, m38))


def seed_family_matrix(pairs, ctx):
    cache = ctx.setdefault("site_cache", {})
    rows = []
    for a, b in pairs:
        a, b = ordered(int(a), int(b))
        if (a, b) not in cache:
            valid = a < H.N_LNC <= b
            c = seed_family_counts(ctx["seq"][a], ctx["seq"][b]) if valid else (0, 0, 0, 0, 0)
            cache[(a, b)] = np.r_[np.log1p(np.asarray(c, dtype=float)), np.log1p(ctx["len"][a])]
        rows.append(cache[(a, b)])
    return np.asarray(rows)


# ------------------------------------------------------------------------------- one fold, all models
def ordered(a, b):
    """lncRNA slot first (indices < N_LNC are lncRNAs); same-type pairs by index."""
    return (a, b) if a <= b else (b, a)


def pair_matrix(pairs, node_kmer, node_len, node_emb):
    comp, emb = [], []
    for a, b in pairs:
        a, b = ordered(int(a), int(b))
        comp.append(np.r_[node_kmer[a], node_kmer[b], np.log1p(node_len[a]), np.log1p(node_len[b])])
        emb.append(np.r_[node_emb[a], node_emb[b]])
    return np.asarray(comp, dtype=np.float32), np.asarray(emb, dtype=np.float32)


def onehot_pairs(pairs, n_nodes):
    rows = np.repeat(np.arange(len(pairs)), 2)
    cols = np.asarray(pairs, dtype=np.int64).ravel()
    return sparse.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(pairs), n_nodes))


def popularity_features(pairs, tdeg):
    a = np.log1p(tdeg[np.asarray(pairs)[:, 0]])
    b = np.log1p(tdeg[np.asarray(pairs)[:, 1]])
    return np.c_[a, b, a * b]


def run_fold(ctx, split, neg_scheme, tr_e, va_e, held_lnc, seed, stored_degrees):
    import lightgbm as lgb
    H.seed_all(seed)
    rng = np.random.default_rng(seed)
    u_tr, v_tr = tr_e[:, 0], tr_e[:, 1]
    edge_index = R.undirected(u_tr.tolist(), v_tr.tolist())
    m_sel, m_test, lnc_sel, lnc_test = R.partition_eval(split, va_e, held_lnc, rng)
    n_sel, n_test = int(m_sel.sum()), int(m_test.sum())
    if n_sel == 0 or n_test == 0:
        return None
    if neg_scheme == "verbatim":
        neg_all = negative_sampling(edge_index=edge_index, num_nodes=H.N_NODES,
                                    num_neg_samples=len(va_e), method="sparse").t().numpy()
        neg_test = neg_all[m_test]
    else:
        s_test = R.make_neg_sampler(neg_scheme, ctx["edge_set"], ctx["deg"], lnc_test,
                                    pos_pairs=va_e[m_test])
        neg_test = s_test(n_test, seed + 12)
    test_stats = getattr(s_test, "stats", None) if neg_scheme != "verbatim" else None
    pos_test = va_e[m_test]

    tdeg = np.zeros(H.N_NODES)
    for a, b in tr_e:
        tdeg[a] += 1
        tdeg[b] += 1
    test_pairs = np.vstack([pos_test, neg_test])
    y_test = np.r_[np.ones(len(pos_test)), np.zeros(len(neg_test))]
    deg_auc = roc_auc_score(y_test, tdeg[test_pairs[:, 0]] + tdeg[test_pairs[:, 1]])
    for tag, stored_degree in stored_degrees:
        if abs(deg_auc - stored_degree) > 1e-9:
            raise SystemExit(f"fold regeneration mismatch ({tag}): degree {deg_auc:.12f} vs stored "
                             f"{stored_degree:.12f}")

    out = {"degree_sum": float(deg_auc),
           "degree_sum_global": float(roc_auc_score(
               y_test, ctx["deg"][test_pairs[:, 0]] + ctx["deg"][test_pairs[:, 1]]))}
    if test_stats is not None:
        out["test_negatives_redrawn"] = int(test_stats["relaxed"])
    train_lnc = set(range(H.N_LNC)) - (set(int(x) for x in held_lnc) if held_lnc else set())
    scopes = [("", set(range(H.N_LNC)), ctx["deg"])]
    if neg_scheme == "swap":
        # the swap draw reads only the training positives, so it has no R1 scope to contrast with
        scopes = [("@trainonly", train_lnc, R._global_degrees(tr_e))]
    elif neg_scheme != "verbatim":
        # R1 drew the TRAINING negatives of the matched schemes with the sampler's GLOBAL degree
        # (computed over all edges, held-out ones included) and, under the entity-disjoint split,
        # over ALL lncRNAs, held-out ones included. A node's held-out edges therefore raise how often
        # it is met as a training negative; for a held-out lncRNA that is all of its degree. A learner
        # can turn this into an anti-predictive bias. "@trainonly" repeats every learned baseline with
        # training negatives drawn over the training lncRNAs only, with degrees counted on the
        # training edges only, which closes the route. Test negatives are untouched.
        scopes.append(("@trainonly", train_lnc, R._global_degrees(tr_e)))
    for suffix, train_lnc, deg_for_train in scopes:
        _fit_learned(out, suffix, ctx, split, neg_scheme, tr_e, test_pairs, y_test, tdeg, edge_index,
                     seed, train_lnc, deg_for_train)
    # zero-parameter seed-site score (no training)
    sites = []
    for a, b in test_pairs:
        a, b = ordered(int(a), int(b))
        valid = a < H.N_LNC <= b
        sites.append(seed_site_count(ctx["seq"][a], ctx["seq"][b]) if valid else 0)
    out["seed_sites"] = float(roc_auc_score(y_test, np.asarray(sites, dtype=float)))
    out["n_test_pos"] = int(len(pos_test))
    out["test_frac_with_site_pos"] = float(np.mean(np.asarray(sites[:len(pos_test)]) > 0))
    out["test_frac_with_site_neg"] = float(np.mean(np.asarray(sites[len(pos_test):]) > 0))
    out["held_lnc_in_train_negatives_frac"] = _held_share(ctx, split, neg_scheme, tr_e, held_lnc, seed)
    return out


def _held_share(ctx, split, neg_scheme, tr_e, held_lnc, seed):
    """Share of R1-scope training negatives that touch a held-out lncRNA (cold split only)."""
    if split != "cold" or neg_scheme in ("verbatim", "swap"):
        return None
    sampler = R.make_neg_sampler(neg_scheme, ctx["edge_set"], ctx["deg"], set(range(H.N_LNC)), pos_pairs=tr_e)
    neg = np.vstack([sampler(len(tr_e), seed + 1000 + ep) for ep in range(1, EPOCHS + 1)])
    held = set(int(x) for x in held_lnc)
    return float(np.mean([int(a) in held for a, _ in neg]))


def _fit_learned(out, suffix, ctx, split, neg_scheme, tr_e, test_pairs, y_test, tdeg, edge_index, seed,
                 train_lnc, deg_for_train):
    import lightgbm as lgb
    u_tr = tr_e[:, 0]
    # training negatives from the scheme DVMnet trains on
    if neg_scheme == "verbatim":
        negs = []
        for ep in range(1, EPOCHS + 1):
            torch.manual_seed(seed + 1000 + ep)
            negs.append(negative_sampling(edge_index=edge_index, num_nodes=H.N_NODES,
                                          num_neg_samples=len(u_tr), method="sparse").t().numpy())
        neg_tr = np.vstack(negs)
    else:
        sampler = R.make_neg_sampler(neg_scheme, ctx["edge_set"], deg_for_train, train_lnc, pos_pairs=tr_e)
        neg_tr = np.vstack([sampler(len(u_tr), seed + 1000 + ep) for ep in range(1, EPOCHS + 1)])
    train_pairs = np.vstack([tr_e, neg_tr])
    y_train = np.r_[np.ones(len(tr_e)), np.zeros(len(neg_tr))]
    w_train = np.r_[np.full(len(tr_e), float(EPOCHS)), np.ones(len(neg_tr))]   # balance classes

    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(popularity_features(train_pairs, tdeg), y_train, sample_weight=w_train)
    out["popularity_lr" + suffix] = float(roc_auc_score(
        y_test, lr.decision_function(popularity_features(test_pairs, tdeg))))

    lr = LogisticRegression(C=1.0, max_iter=5000)
    lr.fit(onehot_pairs(train_pairs, H.N_NODES), y_train, sample_weight=w_train)
    out["node_id_lr" + suffix] = float(roc_auc_score(
        y_test, lr.decision_function(onehot_pairs(test_pairs, H.N_NODES))))

    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(seed_family_matrix(train_pairs, ctx), y_train, sample_weight=w_train)
    out["seed_family_lr" + suffix] = float(roc_auc_score(
        y_test, lr.decision_function(seed_family_matrix(test_pairs, ctx))))

    Ctr, Etr = pair_matrix(train_pairs, ctx["kmer"], ctx["len"], ctx["emb"])
    Cte, Ete = pair_matrix(test_pairs, ctx["kmer"], ctx["len"], ctx["emb"])
    for name, A, B in (("composition_gbm", Ctr, Cte), ("doc2vec_gbm", Etr, Ete)):
        gbm = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                 random_state=seed, verbose=-1, n_jobs=1)
        gbm.fit(A, y_train, sample_weight=w_train)
        out[name + suffix] = float(roc_auc_score(y_test, gbm.predict_proba(B)[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", default=os.path.join(HERE, "dvmnet_revision_results.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "dvmnet_baselines_results.json"))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--sensitivity", default=os.path.join(HERE, "dvmnet_sensitivity_results.json"),
                    help="R1 sensitivity conditions (alpha 0.5, joint, hub-excluded; 3 partitions)")
    ap.add_argument("--r2", default=os.path.join(HERE, "..", "revision_r2", "results", "dvmnet_r2.json"),
                    help="DVMnet re-run with training negatives from the training graph (R2)")
    ap.add_argument("--conditions", default="", help="comma-separated subset, e.g. random-swap,cold-swap")
    args = ap.parse_args()

    t0 = time.time()
    torch.set_num_threads(1)
    H.seed_all(R.SEED)
    feat = H.build_features(embed_cache=os.path.join(HERE, "dvmnet_embed_seed42.npz"))
    seqs, missing = node_sequences(feat)
    print(f"nodes {len(seqs)}; without sequence {missing}", flush=True)
    edges = np.stack(H.edges_as_arrays(feat["node_table"]), axis=1)
    edges = np.stack(H.normalize_edge_orientation(edges[:, 0], edges[:, 1]), axis=1)
    ctx = dict(edge_set={(int(a), int(b)) for a, b in edges}, deg=R._global_degrees(edges),
               seq=seqs, kmer=np.asarray([kmer_frac(s) for s in seqs], dtype=np.float32),
               len=np.asarray([len(s) for s in seqs], dtype=float),
               emb=feat["x_embedding"].numpy())
    r1 = {(c["split"], c["negatives"]): c for c in json.load(open(args.r1))}
    r2 = ({(c["split"], c["negatives"]): c for c in json.load(open(args.r2))}
          if args.r2 and os.path.exists(args.r2) else {})
    plan = [(sp, ng, args.repeats, r1.get((sp, ng))) for sp, ng in CONDITIONS]
    if args.sensitivity and os.path.exists(args.sensitivity):
        plan += [(c["split"], c["negatives"], c["repeats"], c) for c in json.load(open(args.sensitivity))]
    if args.conditions:
        keep = set(args.conditions.split(","))
        plan = [x for x in plan if f"{x[0]}-{x[1]}" in keep]
    print(f"conditions: {', '.join(f'{a}-{b}' for a, b, _, _ in plan)}; DVMnet R2 entries: "
          f"{', '.join(f'{a}-{b}' for a, b in r2) or 'none'}", flush=True)

    results = []
    for split, neg, n_rep, stored in plan:
        s2 = r2.get((split, neg))
        per_repeat = []
        for rep in range(n_rep):
            cv_seed = R.SEED + 100 * rep
            fold_iter = (R.folds_random_edge(edges, cv_seed) if split == "random"
                         else R.folds_cold_start(edges, cv_seed))
            f1 = stored["per_repeat"][rep]["folds"] if stored is not None else None
            f2 = s2["per_repeat"][rep]["folds"] if s2 is not None and rep < s2["repeats"] else None
            folds, j = [], 0
            for k, (tr_e, va_e, _allowed, held) in enumerate(fold_iter):
                if len(va_e) == 0:
                    continue
                checks = ([("R1", f1[j]["degree_test"])] if f1 else []) + \
                         ([("R2", f2[j]["degree_test"])] if f2 else [])
                r = run_fold(ctx, split, neg, tr_e, va_e, held, cv_seed + k, checks)
                if r is None:
                    continue
                if f1:
                    r["DVMnet"] = float(f1[j]["nested"])
                if f2:
                    r["DVMnet@trainonly"] = float(f2[j]["nested"])
                r["regeneration_checked_against"] = [t for t, _ in checks]
                folds.append(r)
                j += 1
            agg = {m: float(np.mean([f[m] for f in folds])) for m in folds[0]
                   if m not in ("n_test_pos", "regeneration_checked_against") and folds[0][m] is not None}
            agg["repeat"] = rep
            agg["folds"] = folds
            per_repeat.append(agg)
            print(f"  {split}-{neg} rep{rep}: " + " | ".join(
                f"{m} {agg[m]:.3f}" for m in ("degree_sum", "degree_sum_global", "popularity_lr",
                                               "popularity_lr@trainonly", "node_id_lr",
                                               "node_id_lr@trainonly", "composition_gbm",
                                               "composition_gbm@trainonly", "doc2vec_gbm",
                                               "doc2vec_gbm@trainonly", "seed_sites", "seed_family_lr",
                                               "seed_family_lr@trainonly", "DVMnet",
                                               "DVMnet@trainonly") if m in agg),
                flush=True)
        models = [m for m in per_repeat[0] if m not in ("repeat", "folds")]
        summary = {}
        for m in models:
            v = np.array([r[m] for r in per_repeat if m in r])
            summary[m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                              values=[float(x) for x in v])
        results.append(dict(split=split, negatives=neg, repeats=n_rep, summary=summary,
                            per_repeat=per_repeat))
    with open(args.out, "w") as fh:
        json.dump(dict(results=results, seconds=round(time.time() - t0, 1),
                       folds_regenerated_and_matched=True,
                       note="every fold's regenerated degree-sum AUROC equals the stored DVMnet run's "
                            "(regeneration_checked_against per fold); conditions without a stored run "
                            "are checked when the tables are assembled"), fh, indent=1)
    print(f"done ({time.time()-t0:.0f}s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
