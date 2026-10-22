#!/usr/bin/env python
"""Target B / DVMnet -- revision experiments for SAGMB.2026.0060 (Reviewer 2, points 1-3).

Extends exp_dvmnet_model.py with three things the reviewer asked for, without changing the
model, its hyperparameters, or the feature construction (all still imported verbatim from
external/DVMnet).

R2-1  NEGATIVE-SAMPLING SENSITIVITY.  The submitted version reported two negative schemes
      (uniform over valid bipartite non-edges; a configuration-model "degree-matched" null with
      endpoints drawn independently with probability proportional to degree).  The reviewer
      objects that the degree-matched null preferentially draws (high-degree, high-degree)
      non-edges, which are the pairs most likely to be UNOBSERVED TRUE interactions, so the
      performance drop could reflect the construction rather than the removal of a degree
      shortcut.  We therefore generalise the sampler to a one-parameter family and add two
      independent constructions:
        alpha<a> : endpoints drawn with probability proportional to degree**a.  a=0 is exactly
                   the uniform scheme, a=1 is exactly the previously reported degree-matched
                   scheme.  A monotone response across a is evidence that the effect tracks the
                   degree shortcut itself and not one particular null.
        joint    : matches the JOINT (binned) endpoint-degree distribution of the evaluation-fold
                   positives, rather than the two marginals independently.  This is the stricter
                   reading of "degree-matched" and is not biased toward hub-hub pairs.
        nohub    : the a=1 scheme with hub-hub pairs excluded -- a negative is rejected when BOTH
                   endpoints fall in the top decile of degree within their own node type.  This
                   isolates exactly the population the reviewer identifies as likely unobserved
                   positives.

R2-2  CHECKPOINT SELECTION.  The submitted version estimated the selection effect as
      (best epoch on the evaluation fold) - (final epoch).  The reviewer correctly notes that the
      final epoch is not a valid reference.  We therefore partition each held-out fold into a
      SELECTION half and a TEST half that are disjoint (for cold-start splits the partition is on
      lncRNA ENTITIES, so the test half stays entity-disjoint from both training and selection),
      choose the epoch on the selection half alone, and report the test half at that epoch.
      Reported per fold:
        sel_best    : max AUROC over epochs on the selection half (what selection sees)
        nested      : AUROC on the TEST half at the epoch chosen on the selection half  <- honest
        test_final  : AUROC on the TEST half at the last epoch
        test_oracle : max AUROC over epochs on the TEST half (selection ON the test half)
      The selection bias is then test_oracle - nested, measured against a valid reference.

R2-3  STATISTICAL INFERENCE.  Folds of a single CV partition share training data, so five folds
      are not five independent observations.  We repeat the whole protocol with independent CV
      partitions and independent negative draws (--repeats), and report the repeat-level
      distribution.  Every per-fold and per-repeat number is written to the JSON artefact so that
      any interval reported in the paper can be recomputed without parsing a log.

Continuity: full_best / full_final reproduce the quantities reported in the submitted version
(max and last-epoch AUROC over the WHOLE held-out fold), so the revision can state both.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import negative_sampling

import dvmnet_harness as H

SEED = 42
EPOCHS = 19            # upstream: `for epoch in range(1, 20)`
LR = 0.0008            # upstream
HIDDEN, OUT = 128, 64  # upstream: DVMNet(num_features, 128, 64)
N_DEG_BINS = 5         # quantile bins for the `joint` scheme
MAX_FOLDS = 0          # smoke tests only (--max-folds); 0 = every fold
HUB_Q = 0.90           # `nohub`: top decile within node type
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ degrees
def _global_degrees(edges):
    deg = np.zeros(H.N_NODES)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def _qbins(vals, nbins):
    """Quantile bin edges that tolerate heavy ties (many zero-degree nodes)."""
    qs = np.linspace(0, 1, nbins + 1)[1:-1]
    edges = np.unique(np.quantile(vals, qs))
    return edges


# ---------------------------------------------------------------- negatives
def make_neg_sampler(scheme, edge_set, deg, allowed_lnc, pos_pairs=None):
    """Return sample(k, seed) -> array of k valid (lnc, mi) non-edges.

    scheme:
      'uniform'            endpoints uniform                      (== alpha0)
      'alpha<a>'           endpoints proportional to degree**a     (a=1 == the submitted 'degmat')
      'degmat'             alias of alpha1
      'joint'              matches the binned JOINT endpoint-degree distribution of pos_pairs
      'nohub'              alpha1, rejecting pairs whose BOTH endpoints are top-decile degree

    Rejection is always against the FULL observed edge set, so a held-out positive is never
    handed to the model as a negative.
    """
    if scheme == "swap":
        if pos_pairs is None or len(pos_pairs) == 0:
            raise ValueError("swap scheme needs the positives it must match")
        return _swap_sampler(np.asarray(pos_pairs), edge_set)

    lnc_arr = np.asarray(sorted(allowed_lnc))
    mi_arr = np.arange(H.N_LNC, H.N_NODES)

    if scheme == "degmat":
        scheme = "alpha1"
    power = None
    if scheme == "uniform":
        power = 0.0
    elif scheme.startswith("alpha"):
        power = float(scheme[5:])
    elif scheme == "nohub":
        power = 1.0

    def _profile(arr, a):
        p = deg[arr].astype(float) ** a
        if p.sum() <= 0:
            p = np.ones_like(p)
        return p / p.sum()

    pl = pm = None
    if power is not None:
        pl = None if power == 0.0 else _profile(lnc_arr, power)
        pm = None if power == 0.0 else _profile(mi_arr, power)

    hub_l = hub_m = None
    if scheme == "nohub":
        hub_l = set(lnc_arr[deg[lnc_arr] >= np.quantile(deg[lnc_arr], HUB_Q)].tolist())
        hub_m = set(mi_arr[deg[mi_arr] >= np.quantile(deg[mi_arr], HUB_Q)].tolist())

    cell_keys = cell_probs = cell_members = None
    if scheme == "joint":
        if pos_pairs is None or len(pos_pairs) == 0:
            raise ValueError("joint scheme needs the positives it must match")
        bl = _qbins(deg[lnc_arr], N_DEG_BINS)
        bm = _qbins(deg[mi_arr], N_DEG_BINS)
        lb = {int(n): int(np.digitize(deg[n], bl)) for n in lnc_arr}
        mb = {int(n): int(np.digitize(deg[n], bm)) for n in mi_arr}
        hist = {}
        for u, v in pos_pairs:
            hist[(lb[int(u)], mb[int(v)])] = hist.get((lb[int(u)], mb[int(v)]), 0) + 1
        members_l, members_m = {}, {}
        for n in lnc_arr:
            members_l.setdefault(lb[int(n)], []).append(int(n))
        for n in mi_arr:
            members_m.setdefault(mb[int(n)], []).append(int(n))
        # keep only cells that can actually be filled
        keys = [k for k in hist if k[0] in members_l and k[1] in members_m]
        if not keys:
            raise ValueError("joint scheme: no fillable degree cell")
        w = np.array([hist[k] for k in keys], dtype=float)
        cell_keys = keys
        cell_probs = w / w.sum()
        cell_members = (members_l, members_m)

    def sample(k, seed):
        r = np.random.default_rng(seed)
        negs = set()
        guard = 0
        limit = max(k * 10000, 100000)
        while len(negs) < k and guard < limit:
            guard += 1
            if scheme == "joint":
                ci = int(r.choice(len(cell_keys), p=cell_probs))
                kl, km = cell_keys[ci]
                u = int(r.choice(cell_members[0][kl]))
                v = int(r.choice(cell_members[1][km]))
            else:
                u = int(r.choice(lnc_arr, p=pl))
                v = int(r.choice(mi_arr, p=pm))
                if scheme == "nohub" and u in hub_l and v in hub_m:
                    continue
            if (u, v) not in edge_set:
                negs.add((u, v))
        if len(negs) < k:
            raise RuntimeError(f"sampler '{scheme}' exhausted: {len(negs)}/{k} after {guard} draws")
        out = sorted(negs)
        r.shuffle(out)
        return np.array(out[:k])

    return sample


def _swap_sampler(pos, edge_set, max_sweeps=500, tries=200, patience=20):
    """R2 'swap': the negatives are the positives' own endpoints, re-paired.

    The miRNA column of the positives is permuted, and every pair that is an observed edge or a
    duplicate is repaired by exchanging its miRNA with that of another pair, accepting an exchange
    only when both resulting pairs are valid. Every lncRNA and every miRNA then appears among the
    negatives exactly as many times as among the positives, so no property of a single node (its
    degree in any graph, its identity, its sequence) can separate the classes; what is left to a
    model is the pairing. It reads nothing but the positives of the set it is drawn for, so a
    training draw cannot see held-out edges and an evaluation draw cannot see the training graph.

    Exactness is impossible when one lncRNA holds so many of a set's positives that the set does not
    contain enough other miRNAs for it (a hub in a small evaluation half). The pairs that no exchange
    can repair keep their lncRNA and take a miRNA drawn from the set's positives (with replacement),
    or failing that from all miRNAs, never a known partner and never a repeat. The lncRNA counts stay
    exact; how many pairs needed this is returned in sample.stats and reported per fold.
    """
    lnc = pos[:, 0].astype(np.int64)
    mi0 = pos[:, 1].astype(np.int64)
    n = len(pos)
    all_mi = np.arange(H.N_LNC, H.N_NODES)

    def sample(k, seed):
        if k != n:
            raise ValueError(f"swap: asked for {k} negatives for {n} positives")
        r = np.random.default_rng(seed)
        mi = mi0[r.permutation(n)].copy()
        cnt = {}
        for p in zip(lnc.tolist(), mi.tolist()):
            cnt[p] = cnt.get(p, 0) + 1

        def bad(i):
            p = (int(lnc[i]), int(mi[i]))
            return p in edge_set or cnt[p] > 1

        best, stale = n + 1, 0
        for _ in range(max_sweeps):
            todo = [i for i in range(n) if bad(i)]
            if not todo:
                break
            if len(todo) < best:
                best, stale = len(todo), 0
            else:
                stale += 1
                if stale >= patience:
                    break
            for i in todo:
                if not bad(i):
                    continue
                for _ in range(tries):
                    j = int(r.integers(n))
                    if j == i or mi[j] == mi[i] or lnc[j] == lnc[i]:
                        continue
                    a, b = (int(lnc[i]), int(mi[j])), (int(lnc[j]), int(mi[i]))
                    if a in edge_set or b in edge_set or cnt.get(a, 0) or cnt.get(b, 0):
                        continue
                    cnt[(int(lnc[i]), int(mi[i]))] -= 1
                    cnt[(int(lnc[j]), int(mi[j]))] -= 1
                    mi[i], mi[j] = mi[j], mi[i]
                    cnt[a] = 1
                    cnt[b] = 1
                    break
        relaxed = outside = 0
        for i in [i for i in range(n) if bad(i)]:
            u = int(lnc[i])
            ok = [int(m) for m in mi0 if (u, int(m)) not in edge_set and not cnt.get((u, int(m)), 0)]
            if not ok:
                ok = [int(m) for m in all_mi if (u, int(m)) not in edge_set and not cnt.get((u, int(m)), 0)]
                outside += 1
            if not ok:
                raise RuntimeError(f"swap: lncRNA {u} has no admissible miRNA left")
            cnt[(u, int(mi[i]))] -= 1
            mi[i] = ok[int(r.integers(len(ok)))]
            cnt[(u, int(mi[i]))] = 1
            relaxed += 1
        assert not any(bad(i) for i in range(n))
        sample.stats = dict(n=n, relaxed=relaxed, outside_pool=outside)
        return np.stack([lnc, mi], axis=1)

    sample.stats = None
    return sample


# --------------------------------------------------------------- data/splits
def undirected(u, v):
    ui = [x for pair in zip(u, v) for x in pair]
    vi = [x for pair in zip(v, u) for x in pair]
    return torch.stack([torch.tensor(ui), torch.tensor(vi)], dim=0)


def folds_random_edge(edges, cv_seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    for tr, va in kf.split(edges):
        yield edges[tr], edges[va], set(range(H.N_LNC)), None


def folds_cold_start(edges, cv_seed):
    """Entity-disjoint on lncRNA nodes: every held-out lncRNA has zero degree in the train graph."""
    lnc_ids = np.arange(H.N_LNC)
    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    for tr_l, va_l in kf.split(lnc_ids):
        held = set(lnc_ids[va_l].tolist())
        mask = np.array([u in held for u, _ in edges])
        yield edges[~mask], edges[mask], held, held


def partition_eval(split, va_e, held_lnc, rng):
    """Split the held-out fold into a SELECTION half and a TEST half.

    random split : partition the held-out EDGES at random.
    cold split   : partition the held-out lncRNA ENTITIES, so the test half remains
                   entity-disjoint from both the training graph and the selection half.
    Returns (mask_sel, mask_test, allowed_lnc_sel, allowed_lnc_test).
    """
    n = len(va_e)
    if split == "random":
        idx = rng.permutation(n)
        cut = n // 2
        m_sel = np.zeros(n, dtype=bool)
        m_sel[idx[:cut]] = True
        return m_sel, ~m_sel, set(range(H.N_LNC)), set(range(H.N_LNC))
    ent = np.array(sorted(held_lnc))
    ent = ent[rng.permutation(len(ent))]
    cut = len(ent) // 2
    sel_ent, test_ent = set(ent[:cut].tolist()), set(ent[cut:].tolist())
    m_sel = np.array([int(u) in sel_ent for u, _ in va_e])
    m_test = np.array([int(u) in test_ent for u, _ in va_e])
    return m_sel, m_test, sel_ent, test_ent


# -------------------------------------------------------------------- runner
def run_fold(feat, split, neg_scheme, tr_e, va_e, allowed_lnc, held_lnc,
             edge_set, deg, seed, verbose=False, train_scope="global"):
    """train_scope (R2): which degrees and lncRNAs the TRAINING negatives are drawn with.

    'global' is the R1 run: degrees over all edges, held-out ones included, and (cold split) all
    lncRNAs, held-out ones included. A node's held-out edges then raise how often it is met as a
    training negative -- for a held-out lncRNA that is all of its degree -- which a learner can turn
    into an anti-predictive bias. 'train' draws them with degrees counted on the training edges and
    over the training lncRNAs only. The evaluation negatives are the same in both.
    """
    from model import DVMNet

    H.seed_all(seed)
    rng = np.random.default_rng(seed)
    u_tr, v_tr = tr_e[:, 0], tr_e[:, 1]
    u_va, v_va = va_e[:, 0], va_e[:, 1]
    edge_index = undirected(u_tr.tolist(), v_tr.tolist())

    m_sel, m_test, lnc_sel, lnc_test = partition_eval(split, va_e, held_lnc, rng)
    n_sel, n_test = int(m_sel.sum()), int(m_test.sum())
    if n_sel == 0 or n_test == 0:
        return None

    # ---- negatives, drawn separately for the two halves so a cold-start test half never
    #      receives a negative touching a selection-half entity
    if neg_scheme == "verbatim":
        neg_all = negative_sampling(
            edge_index=edge_index, num_nodes=H.N_NODES, num_neg_samples=len(u_va),
            method="sparse",
        ).t().numpy()
        neg_sel, neg_test = neg_all[m_sel], neg_all[m_test]
    else:
        pos_sel = va_e[m_sel]
        pos_test = va_e[m_test]
        s_sel = make_neg_sampler(neg_scheme, edge_set, deg, lnc_sel, pos_pairs=pos_sel)
        s_test = make_neg_sampler(neg_scheme, edge_set, deg, lnc_test, pos_pairs=pos_test)
        neg_sel = s_sel(n_sel, seed + 11)
        neg_test = s_test(n_test, seed + 12)
        neg_all = np.vstack([neg_sel, neg_test])

    def _impossible(neg):
        return sum(1 for a, b in neg
                   if (a < H.N_LNC and b < H.N_LNC) or (a >= H.N_LNC and b >= H.N_LNC)) / len(neg)

    # ---- label vectors for the three evaluation sets
    def _eval_set(pos, neg):
        eli = torch.cat([torch.tensor(pos.T), torch.tensor(neg.T)], dim=-1)
        lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        return eli, lab

    eli_sel, lab_sel = _eval_set(va_e[m_sel], neg_sel)
    eli_test, lab_test = _eval_set(va_e[m_test], neg_test)
    eli_full, lab_full = _eval_set(va_e, neg_all)

    # ---- zero-parameter degree-sum baseline on the identical test half
    tdeg = np.zeros(H.N_NODES)
    for a, b in tr_e:
        tdeg[a] += 1
        tdeg[b] += 1
    s_test_scores = np.r_[
        [tdeg[a] + tdeg[b] for a, b in va_e[m_test]],
        [tdeg[a] + tdeg[b] for a, b in neg_test],
    ]
    degree_test = roc_auc_score(lab_test, s_test_scores)
    degree_test_global = roc_auc_score(lab_test, np.r_[
        [deg[a] + deg[b] for a, b in va_e[m_test]], [deg[a] + deg[b] for a, b in neg_test]])

    # ---- build Data objects exactly as upstream main.py does
    train_data = Data(
        x=feat["x_embedding"], x_lnc=feat["x_lnc"], x_mi=feat["x_mi"],
        edge_index=edge_index, edge_label=torch.ones(len(u_tr)),
        edge_label_index=torch.stack([torch.tensor(u_tr), torch.tensor(v_tr)], dim=0),
        xname=feat["graph_label"], di=feat["ser_di"].values, sub=feat["mi_sub"].values,
    )
    base_val = Data(
        x=feat["x_embedding"], edge_index=edge_index,
        edge_label=torch.zeros(1), edge_label_index=eli_full,
        xname=feat["graph_label"], di=feat["ser_di"].values, sub=feat["mi_sub"].values,
    )
    tf = NormalizeFeatures()
    train_data, base_val = tf(train_data), tf(base_val)

    model = DVMNet(train_data.num_features, HIDDEN, OUT)
    opt = torch.optim.Adam(params=model.parameters(), lr=LR)
    crit = torch.nn.BCEWithLogitsLoss()

    if train_scope == "global":
        deg_train, lnc_train = deg, set(range(H.N_LNC))
    elif train_scope == "train":
        deg_train = _global_degrees(tr_e)
        lnc_train = set(range(H.N_LNC)) - (set(int(x) for x in held_lnc) if held_lnc else set())
    else:
        raise ValueError(train_scope)
    train_sampler = (
        None if neg_scheme == "verbatim"
        else make_neg_sampler(neg_scheme, edge_set, deg_train, lnc_train, pos_pairs=tr_e)
    )

    auc_sel, auc_test, auc_full = [], [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        if train_sampler is None:
            nidx = negative_sampling(
                edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=train_data.edge_label_index.size(1), method="sparse",
            )
        else:
            nidx = torch.tensor(train_sampler(len(u_tr), seed + 1000 + epoch).T)
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
            scores = {}
            for tag, eli_e in (("sel", eli_sel), ("test", eli_test), ("full", eli_full)):
                vout, *_ = model.mainNet(
                    base_val.x, train_data.x_lnc, train_data.x_mi, base_val.edge_index,
                    base_val.xname, eli_e, base_val.di, base_val.sub,
                )
                scores[tag] = vout.view(-1).numpy()
        auc_sel.append(roc_auc_score(lab_sel, scores["sel"]))
        auc_test.append(roc_auc_score(lab_test, scores["test"]))
        auc_full.append(roc_auc_score(lab_full, scores["full"]))
        if verbose:
            print(f"      ep{epoch:02d} loss {loss.item():.4f} sel {auc_sel[-1]:.4f} "
                  f"test {auc_test[-1]:.4f}", flush=True)

    chosen = int(np.argmax(auc_sel))
    return dict(
        n_val_pos=int(len(u_va)), n_sel_pos=n_sel, n_test_pos=n_test,
        chosen_epoch=chosen + 1,
        sel_best=float(auc_sel[chosen]),
        nested=float(auc_test[chosen]),           # honest: chosen on sel, scored on test
        test_final=float(auc_test[-1]),
        test_oracle=float(max(auc_test)),
        full_best=float(max(auc_full)),           # continuity with the submitted version
        full_final=float(auc_full[-1]),
        degree_test=float(degree_test),
        degree_test_global=float(degree_test_global),
        impossible_frac=float(_impossible(neg_all)),
        train_scope=train_scope,
    )


def run_condition(feat, split, neg_scheme, repeats, verbose=False, train_scope="global"):
    edges = np.stack(H.edges_as_arrays(feat["node_table"]), axis=1)
    edges = np.stack(H.normalize_edge_orientation(edges[:, 0], edges[:, 1]), axis=1)
    edge_set = {(int(a), int(b)) for a, b in edges}
    deg = _global_degrees(edges)

    per_repeat = []
    for rep in range(repeats):
        cv_seed = SEED + 100 * rep
        fold_iter = (folds_random_edge(edges, cv_seed) if split == "random"
                     else folds_cold_start(edges, cv_seed))
        folds = []
        for k, (tr_e, va_e, allowed_lnc, held) in enumerate(fold_iter):
            if len(va_e) == 0:
                continue
            if MAX_FOLDS and len(folds) >= MAX_FOLDS:
                break
            r = run_fold(feat, split, neg_scheme, tr_e, va_e, allowed_lnc, held,
                         edge_set, deg, seed=cv_seed + k, verbose=verbose, train_scope=train_scope)
            if r is None:
                continue
            folds.append(r)
            print(f"    rep{rep} fold{k}: nested {r['nested']:.4f} | oracle {r['test_oracle']:.4f}"
                  f" | test_final {r['test_final']:.4f} | full_best {r['full_best']:.4f}"
                  f" | deg {r['degree_test']:.4f} | ep* {r['chosen_epoch']}", flush=True)
        agg = {k: float(np.mean([f[k] for f in folds]))
               for k in ("nested", "test_oracle", "test_final", "full_best", "full_final",
                         "degree_test", "degree_test_global", "impossible_frac", "sel_best")}
        agg["selection_bias"] = agg["test_oracle"] - agg["nested"]
        agg["repeat"] = rep
        agg["folds"] = folds
        per_repeat.append(agg)
        print(f"  rep{rep} MEAN: nested {agg['nested']:.4f} | oracle {agg['test_oracle']:.4f}"
              f" | bias {agg['selection_bias']:+.4f} | degree {agg['degree_test']:.4f}", flush=True)

    def across(key):
        v = np.array([r[key] for r in per_repeat])
        return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    lo=float(v.min()), hi=float(v.max()), values=[float(x) for x in v])

    return dict(
        split=split, negatives=neg_scheme, repeats=repeats, train_scope=train_scope,
        summary={k: across(k) for k in
                 ("nested", "test_oracle", "test_final", "full_best", "full_final",
                  "degree_test", "degree_test_global", "selection_bias", "impossible_frac")},
        per_repeat=per_repeat,
    )


def main():
    global EPOCHS, MAX_FOLDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", required=True,
                    help="comma-separated <split>-<negscheme>, e.g. cold-alpha0.5,random-joint")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=os.path.join(HERE, "dvmnet_revision_results.json"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--train-scope", choices=("global", "train"), default="global",
                    help="R2: 'train' draws the training negatives from the training graph only")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch threads (0 = library default, as in the R1 run)")
    ap.add_argument("--max-folds", type=int, default=0, help="smoke test: stop after this many folds")
    args = ap.parse_args()
    EPOCHS = args.epochs
    MAX_FOLDS = args.max_folds
    if args.threads:
        torch.set_num_threads(args.threads)

    H.seed_all(SEED)
    feat = H.build_features(embed_cache=os.path.join(HERE, "dvmnet_embed_seed42.npz"))

    results = []
    if os.path.exists(args.out):                      # resume-friendly
        try:
            results = json.load(open(args.out))
            print(f"resuming: {len(results)} condition(s) already in {args.out}", flush=True)
        except Exception:
            results = []
    done = {(r["condition"]) for r in results}

    for name in args.conditions.split(","):
        if name in done:
            print(f"skip {name} (already present)", flush=True)
            continue
        split, neg = name.split("-", 1)
        assert split in ("random", "cold"), name
        print(f"\n=== {name} (split={split}, negatives={neg}, repeats={args.repeats}) ===",
              flush=True)
        t0 = time.time()
        r = run_condition(feat, split, neg, args.repeats, verbose=args.verbose,
                          train_scope=args.train_scope)
        r["condition"] = name
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        s = r["summary"]
        print(f"  -> nested {s['nested']['mean']:.4f} ± {s['nested']['sd']:.4f} | "
              f"oracle {s['test_oracle']['mean']:.4f} | "
              f"selection bias {s['selection_bias']['mean']:+.4f} ± {s['selection_bias']['sd']:.4f} | "
              f"degree {s['degree_test']['mean']:.4f} | {r['seconds']}s", flush=True)
        with open(args.out, "w") as f:                # write after every condition
            json.dump(results, f, indent=2)
        print(f"  wrote {args.out} ({len(results)} conditions)", flush=True)


if __name__ == "__main__":
    main()
