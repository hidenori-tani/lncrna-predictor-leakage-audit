#!/usr/bin/env python
"""Target C / LncPTPred -- reproduce the PUBLISHED model on the FULL benchmark with the authors' own
training settings, and only then ask how its score depends on evaluation choices.

Reviewer 2 (R1 round), point 3: "Ideally, the authors should reproduce the model using the full
dataset and the original training settings, where available, before drawing conclusions about its
dependence on particular evaluation choices."

The R1 analysis said the published record supplies no training protocol. That was wrong: the
released repository (github.com/zglabDIB/lncptpred @ 621391287402) ships `models/models.zip`, the
trained model itself. Read without executing it (revision_r2/inspect_released_lncptpred_model.py),
its pickled state records

  * the sklearn settings: num_leaves 204, max_depth 14, learning_rate 0.0344408, n_estimators 20000,
    min_child_samples 13, colsample_bytree 0.7274, reg_alpha 3.0703, reg_lambda 3.0936,
    subsample 0.5724 with subsample_freq 0 (so no bagging), subsample_for_bin 250000;
  * an eval record with two sets, "training" and "valid_1", and best_iteration 10,596 out of 10,646
    recorded (= early stopping on valid_1 binary_logloss with a patience of 50);
  * 1,272,014 training rows (root internal_count of Tree 0, and the scaler's n_samples_seen_),
    i.e. an 80/20 split; valid_1 logloss at the selected iteration = 0.48258.

That last number is the first numeric target this benchmark has ever had, and it is the authors' own.

Stages (run one at a time; each writes its own JSON):
  features        all 1,590,016 rows x 720 features, in the RELEASED column order
                  (9 structure, mfe, 340 k-mer fractions, Strand | 369 protein), stored as float16
                  in a memmap -- the released pipeline casts to float16 before scaling.
  scaler-check    our column means / SDs against the released scaler's 720 means / scales.
  score-released  the released model on every row (float16 -> scaler -> float16 -> predict), and the
                  rows-per-leaf of its first trees against the leaf_count the model stores.
  retrain         the authors' settings on an 80/20 split stratified by protein x label (the paper's
                  stated stratification), validation split cut into halves A and B; every iteration's
                  logloss is recorded on A and B so that "stop on the whole validation split" (what the
                  authors did) and "stop on A, report B" (independent selection) come from ONE fit.
  r1-settings     the R1 reconstruction's settings (200 trees, LightGBM defaults) on the same split.
  sub400k         the authors' settings on the R1 subsample size, to separate data size from settings.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lncptpred_exact as E   # the verbatim ports of the released feature code (R1)

ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "external", "LncPTPred", "dataset")
WORK = os.path.join(HERE, "lncptpred_full")                      # derived, git-ignored
RELEASED = os.path.join(ROOT, "external", "LncPTPred_released_models", "derived")   # git-ignored
SEED = 42
N_JOBS = 4            # half the logical cores: the machine is shared with the author's own work
CHUNK = 100_000
N_FEAT = 720

# the released model's constructor settings (decoded from its pickled state; see docstring)
ORIGINAL = dict(boosting_type="gbdt", objective="binary", num_leaves=204, max_depth=14,
                learning_rate=0.03444081993284159, min_child_samples=13, min_child_weight=0.001,
                min_split_gain=0.0, colsample_bytree=0.7274007249599967, reg_alpha=3.070346559732223,
                reg_lambda=3.0935935089398225, subsample=0.572447599448142, subsample_freq=0,
                subsample_for_bin=250000)
ORIGINAL_MAX_ROUNDS = 20000
ORIGINAL_PATIENCE = 50          # 10,646 recorded - 10,596 best
ORIGINAL_VALID_LOGLOSS = 0.4825826409413437
ORIGINAL_TRAIN_LOGLOSS = 0.20722714949259616
ORIGINAL_N_TRAIN = 1272014


def _paths():
    return dict(X=os.path.join(WORK, "X_f16.npy"), meta=os.path.join(WORK, "meta.npz"),
                struct=os.path.join(WORK, "struct_all.npz"))


def load_table():
    df = pd.read_csv(os.path.join(DATA, "Final_lncRNA_Protein_Interaction.txt"), sep="\t")
    return df


# ------------------------------------------------------------------------------ stage: features
def stage_features(args):
    os.makedirs(WORK, exist_ok=True)
    P = _paths()
    df = load_table()
    n = len(df)
    seqs = df["Sequence"].astype(str).tolist()
    bad = sum(1 for s in seqs if re.search(r"[^ACGT]", s))
    print(f"rows {n:,}; sequences with a non-ACGT character: {bad:,}", flush=True)

    prot, prot_cols = E.protein_features()
    assert len(prot_cols) == 369, len(prot_cols)
    pidx = {p: i for i, p in enumerate(prot.index)}
    prot_np = prot.to_numpy(dtype=np.float64)
    pid = df["Protein"].map(pidx).to_numpy()
    assert not np.isnan(pid.astype(float)).any(), "a protein in the table has no sequence"
    pid = pid.astype(np.int16)

    S = E.structure_block(seqs, cache=P["struct"])                  # (n, 10) = 9 metrics + mfe
    strand = df["Strand"].map({"+": 1, "-": -1}).to_numpy(dtype=np.float64)
    assert not np.isnan(strand).any()

    X = np.lib.format.open_memmap(P["X"], mode="w+", dtype=np.float16, shape=(n, N_FEAT))
    t0 = time.time()
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        K = E.kmer_fractions(seqs[a:b])                               # (m, 340)
        block = np.concatenate([S[a:b, :9].astype(np.float64), S[a:b, 9:10].astype(np.float64),
                                K.astype(np.float64), strand[a:b, None],
                                prot_np[pid[a:b]]], axis=1)
        assert block.shape[1] == N_FEAT
        X[a:b] = block.astype(np.float16)
        print(f"  rows {b:,}/{n:,}  ({time.time()-t0:.0f}s)", flush=True)
    X.flush()
    del X

    y = df["Target"].to_numpy(dtype=np.int8)
    gc = np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs], dtype=np.float32)
    length = np.array([len(s) for s in seqs], dtype=np.int16)
    np.savez_compressed(P["meta"], y=y, pid=pid, gc=gc, length=length,
                        proteins=np.array(prot.index, dtype=str), n=n)
    out = dict(n_rows=n, n_features=N_FEAT, non_acgt_sequences=bad,
               column_order="9 structure | mfe | 340 k-mer fractions | Strand | 369 protein",
               dtype="float16", seconds=round(time.time() - t0, 1))
    _dump(out, args.out)


# --------------------------------------------------------------------------- stage: scaler-check
def column_moments(X, rows=None):
    """Mean and SD (ddof=0, as StandardScaler) of each column, accumulated in float64 by chunk."""
    n_all = X.shape[0]
    idx = np.arange(n_all) if rows is None else np.sort(rows)
    s1 = np.zeros(N_FEAT)
    s2 = np.zeros(N_FEAT)
    for a in range(0, len(idx), CHUNK):
        blk = X[idx[a:a + CHUNK]].astype(np.float64)
        s1 += blk.sum(0)
        s2 += (blk * blk).sum(0)
    m = s1 / len(idx)
    return m, np.sqrt(np.maximum(s2 / len(idx) - m * m, 0.0))


def stage_scaler_check(args):
    P = _paths()
    X = np.load(P["X"], mmap_mode="r")
    mean_rel = np.load(os.path.join(RELEASED, "scaler_mean_.npy"))
    scale_rel = np.load(os.path.join(RELEASED, "scaler_scale_.npy"))
    m, sd = column_moments(X)
    blocks = {"structure_9": slice(0, 9), "mfe": slice(9, 10), "kmer_340": slice(10, 350),
              "strand": slice(350, 351), "protein_369": slice(351, 720)}
    res = {}
    for name, sl in blocks.items():
        dm = np.abs(m[sl] - mean_rel[sl])
        z = dm / np.where(scale_rel[sl] > 0, scale_rel[sl], 1.0)      # difference in released SDs
        rs = np.abs(sd[sl] / np.where(scale_rel[sl] > 0, scale_rel[sl], 1.0) - 1.0)
        res[name] = dict(max_abs_mean_diff=float(dm.max()), max_mean_diff_in_sd=float(z.max()),
                         median_mean_diff_in_sd=float(np.median(z)), max_rel_sd_diff=float(rs.max()))
    # the same comparison with the columns in the R1 order (340 k-mer | 10 structure | Strand | 369):
    # if our order were wrong, this is the order that would fit instead
    r1 = np.r_[np.arange(10, 350), np.arange(0, 10), [350], np.arange(351, 720)]
    z_r1 = np.abs(m[r1] - mean_rel) / np.where(scale_rel > 0, scale_rel, 1.0)
    z_ours = np.abs(m - mean_rel) / np.where(scale_rel > 0, scale_rel, 1.0)
    out = dict(blocks=res, max_mean_diff_in_sd_released_order=float(z_ours.max()),
               max_mean_diff_in_sd_R1_order=float(z_r1.max()),
               note="moments over all released rows vs the scaler's moments over the authors' "
                    "1,272,014 training rows; agreement to a few hundredths of an SD is what an "
                    "80% subsample of the same features gives")
    _dump(out, args.out)


# ------------------------------------------------------------------------ stage: score-released
def scaled_batch(X, rows, mean, scale):
    """The released inference path: float16 features -> StandardScaler -> float16."""
    blk = X[rows].astype(np.float64)
    return ((blk - mean) / scale).astype(np.float16).astype(np.float32)


def stage_score_released(args):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, log_loss
    P = _paths()
    X = np.load(P["X"], mmap_mode="r")
    meta = np.load(P["meta"])
    y = meta["y"]
    mean = np.load(os.path.join(RELEASED, "scaler_mean_.npy"))
    scale = np.load(os.path.join(RELEASED, "scaler_scale_.npy"))
    t0 = time.time()
    bst = lgb.Booster(model_file=os.path.join(RELEASED, "LightGBM_lncRNA_Protein.model.txt"))
    print(f"released model loaded: {bst.num_trees():,} trees ({time.time()-t0:.0f}s)", flush=True)
    n = X.shape[0] if args.limit is None else min(args.limit, X.shape[0])

    prob = np.zeros(n, dtype=np.float64)
    leaf0 = np.zeros((n, args.leaf_trees), dtype=np.int32)
    t0 = time.time()
    for a in range(0, n, args.batch):
        b = min(a + args.batch, n)
        xb = scaled_batch(X, np.arange(a, b), mean, scale)
        prob[a:b] = bst.predict(xb, num_threads=N_JOBS)
        leaf0[a:b] = bst.predict(xb, pred_leaf=True, num_iteration=args.leaf_trees,
                                 num_threads=N_JOBS)
        print(f"  scored {b:,}/{n:,}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(WORK, "released_prob.npy" if args.limit is None else "released_prob_head.npy"),
            prob)

    # rows-per-leaf against the leaf_count stored in the model (training rows only)
    dump = bst.dump_model(num_iteration=args.leaf_trees)
    leaf_checks = []
    for t in range(args.leaf_trees):
        stored = {}

        def walk(node):
            if "leaf_index" in node:
                stored[node["leaf_index"]] = node["leaf_count"]
            else:
                walk(node["left_child"])
                walk(node["right_child"])
        walk(dump["tree_info"][t]["tree_structure"])
        ours = np.bincount(leaf0[:, t], minlength=max(stored) + 1)
        st = np.array([stored[k] for k in sorted(stored)], dtype=float)
        ob = ours[sorted(stored)].astype(float)
        ratio = st / np.where(ob > 0, ob, np.nan)
        big = st >= 1000
        leaf_checks.append(dict(tree=t, n_leaves=len(st), stored_total=int(st.sum()),
                                ours_total=int(ob.sum()),
                                ratio_overall=float(st.sum() / ob.sum()),
                                ratio_median_leaves_ge1000=float(np.nanmedian(ratio[big])) if big.any() else None,
                                ratio_p05_p95_leaves_ge1000=[float(np.nanpercentile(ratio[big], 5)),
                                                             float(np.nanpercentile(ratio[big], 95))]
                                if big.any() else None,
                                leaves_with_more_stored_than_ours=int(np.sum(st > ob))))
    yy = y[:n]
    p = np.clip(prob, 1e-15, 1 - 1e-15)
    out = dict(n_scored=int(n), auroc_all_rows=float(roc_auc_score(yy, prob)),
               logloss_all_rows=float(log_loss(yy, p)),
               expected_logloss_if_features_match=float(
                   (ORIGINAL_N_TRAIN * ORIGINAL_TRAIN_LOGLOSS +
                    (len(y) - ORIGINAL_N_TRAIN) * ORIGINAL_VALID_LOGLOSS) / len(y)),
               leaf_count_checks=leaf_checks, seconds=round(time.time() - t0, 1))
    _dump(out, args.out)


# ----------------------------------------------------------------------- shared: data sequences
def make_sequence_class():
    import lightgbm as lgb

    class F16Rows(lgb.Sequence):
        """Rows of the float16 memmap, scaled on the fly, so the full matrix is never held as
        float32 (1.59 M x 720 x 4 B = 4.6 GB would not fit beside the author's own work)."""
        def __init__(self, X, rows, mean, scale, batch_size=65536):
            self.X, self.rows, self.mean, self.scale = X, np.sort(rows), mean, scale
            self.batch_size = batch_size

        def __getitem__(self, i):
            # A slice (the batches LightGBM pushes) stays float32. A single row is what LightGBM
            # samples to place the bin boundaries, and its sampler accepts only float64; the values
            # are the same (float16 -> float32 -> float64 is exact).
            r = self.rows[i]
            return scaled_batch(self.X, np.atleast_1d(r), self.mean, self.scale) if np.ndim(r) else \
                scaled_batch(self.X, np.array([r]), self.mean, self.scale)[0].astype(np.float64)

        def __len__(self):
            return len(self.rows)
    return F16Rows


def split_80_20(meta, seed=SEED, n_sub=None):
    """80/20 split stratified by protein x label (the paper: "stratified by both protein name and
    the target interaction label"); the 20% is cut again into halves A and B, same strata."""
    from sklearn.model_selection import train_test_split
    y, pid = meta["y"], meta["pid"]
    rows = np.arange(len(y))
    strata = pid.astype(np.int64) * 2 + y
    if n_sub is not None:
        rows, _ = train_test_split(rows, train_size=n_sub, stratify=strata, random_state=seed)
        strata = strata[rows]
    tr, va = train_test_split(rows, test_size=0.2, stratify=strata, random_state=seed)
    va_strata = (pid[va].astype(np.int64) * 2 + y[va])
    va_a, va_b = train_test_split(va, test_size=0.5, stratify=va_strata, random_state=seed + 1)
    return np.sort(tr), np.sort(va_a), np.sort(va_b)


def fit_scaler(X, tr):
    m, sd = column_moments(X, tr)
    sd = np.where(sd > 0, sd, 1.0)        # StandardScaler leaves zero-variance columns unscaled
    return m, sd


def replay_early_stopping(curve, patience):
    """LightGBM's rule for a minimised metric: best updates on a strict improvement; stop once
    `patience` iterations pass without one. Returns (best_iteration_1based, stop_iteration_1based)."""
    best, best_i = np.inf, 0
    for i, v in enumerate(curve):
        if v < best:
            best, best_i = v, i
        elif i - best_i >= patience:
            return best_i + 1, i + 1
    return best_i + 1, len(curve)


def predict_rows(bst, X, rows, mean, scale, num_iteration=None, raw=False, start_iteration=0):
    out = np.zeros(len(rows))
    for a in range(0, len(rows), CHUNK):
        r = rows[a:a + CHUNK]
        out[a:a + CHUNK] = bst.predict(scaled_batch(X, r, mean, scale), num_iteration=num_iteration,
                                       start_iteration=start_iteration, raw_score=raw,
                                       num_threads=N_JOBS)
    return out


# ------------------------------------------------------------------------------ stage: retrain
def stage_retrain(args, settings="original", n_sub=None):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, log_loss
    P = _paths()
    X = np.load(P["X"], mmap_mode="r")
    meta = np.load(P["meta"])
    y = meta["y"].astype(np.float64)
    tr, va_a, va_b = split_80_20(meta, n_sub=n_sub)
    va = np.sort(np.r_[va_a, va_b])
    print(f"train {len(tr):,} | valid {len(va):,} (A {len(va_a):,} + B {len(va_b):,})", flush=True)
    mean, scale = fit_scaler(X, tr)
    F16Rows = make_sequence_class()

    if settings == "original":
        params = dict(ORIGINAL)
        rounds, patience = ORIGINAL_MAX_ROUNDS, ORIGINAL_PATIENCE
    else:                                  # the R1 reconstruction: LGBMClassifier(n_estimators=200)
        params = dict(boosting_type="gbdt", objective="binary")
        rounds, patience = 200, None
    params.update(dict(num_threads=N_JOBS, verbose=-1, metric="binary_logloss",
                       bin_construct_sample_cnt=params.pop("subsample_for_bin", 200000)))
    # The released model set no seed (random_state=None): its parameter block shows LightGBM's
    # default seeds (data 1, feature_fraction 2, bagging 3, drop 4), which 4.6 shares, so no seed is
    # passed for the original settings. The R1 reconstruction used random_state=42; kept for it.
    seed_kw = {} if settings == "original" else {"seed": SEED}
    params.update(seed_kw)
    # sklearn-alias -> core names, so that lgb.train sees exactly what LGBMClassifier passed
    alias = dict(min_child_samples="min_data_in_leaf", min_child_weight="min_sum_hessian_in_leaf",
                 min_split_gain="min_gain_to_split", colsample_bytree="feature_fraction",
                 reg_alpha="lambda_l1", reg_lambda="lambda_l2", subsample="bagging_fraction",
                 subsample_freq="bagging_freq", boosting_type="boosting")
    params = {alias.get(k, k): v for k, v in params.items()}

    t0 = time.time()
    dtr = lgb.Dataset(F16Rows(X, tr, mean, scale), label=y[tr], free_raw_data=True,
                      params={"max_bin": 255, "bin_construct_sample_cnt":
                              params["bin_construct_sample_cnt"], "verbose": -1, **seed_kw})
    da = lgb.Dataset(F16Rows(X, va_a, mean, scale), label=y[va_a], reference=dtr)
    db = lgb.Dataset(F16Rows(X, va_b, mean, scale), label=y[va_b], reference=dtr)
    print(f"datasets built ({time.time()-t0:.0f}s)", flush=True)

    curves = {"A": [], "B": []}
    nA, nB = len(va_a), len(va_b)

    def record(env):
        for name, metric, val, _ in env.evaluation_result_list:
            curves[name].append(val)
        it = env.iteration + 1
        if it % 250 == 0:
            union = (nA * curves["A"][-1] + nB * curves["B"][-1]) / (nA + nB)
            print(f"  it {it:>6} | A {curves['A'][-1]:.5f} | B {curves['B'][-1]:.5f} | "
                  f"union {union:.5f} | {time.time()-t0:.0f}s", flush=True)
        if patience is not None:
            u = (nA * np.array(curves["A"]) + nB * np.array(curves["B"])) / (nA + nB)
            done = all(replay_early_stopping(c, patience)[1] < it
                       for c in (u, np.array(curves["A"]), np.array(curves["B"])))
            if done:
                raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

    record.order = 30
    bst = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[da, db], valid_names=["A", "B"],
                    callbacks=[record])
    fit_s = time.time() - t0
    tag = settings if n_sub is None else f"{settings}_n{n_sub}"
    bst.save_model(os.path.join(WORK, f"retrain_{tag}.model.txt"))

    cA, cB = np.array(curves["A"]), np.array(curves["B"])
    cU = (nA * cA + nB * cB) / (nA + nB)
    out = dict(settings=settings, params=params, n_train=int(len(tr)), n_valid=int(len(va)),
               n_A=int(nA), n_B=int(nB), iterations_run=int(len(cA)), fit_seconds=round(fit_s, 1))
    if patience is not None:
        kU, sU = replay_early_stopping(cU, patience)
        kA, sA = replay_early_stopping(cA, patience)
        kB, sB = replay_early_stopping(cB, patience)
        out["early_stopping"] = dict(union=dict(best=kU, stop=sU, logloss=float(cU[kU - 1])),
                                     A=dict(best=kA, stop=sA, logloss=float(cA[kA - 1])),
                                     B=dict(best=kB, stop=sB, logloss=float(cB[kB - 1])))
        ks = sorted({kU, kA, kB})
    else:
        ks = [len(cA)]
        out["valid_logloss_final"] = float(cU[-1])

    # AUROC on A, B and the whole validation split at each selected iteration
    pa = {k: predict_rows(bst, X, va_a, mean, scale, num_iteration=k) for k in ks}
    pb = {k: predict_rows(bst, X, va_b, mean, scale, num_iteration=k) for k in ks}
    auc = {}
    for k in ks:
        pu = np.r_[pa[k], pb[k]]
        yu = np.r_[y[va_a], y[va_b]]
        auc[str(k)] = dict(A=float(roc_auc_score(y[va_a], pa[k])), B=float(roc_auc_score(y[va_b], pb[k])),
                           union=float(roc_auc_score(yu, pu)),
                           union_logloss=float(log_loss(yu, np.clip(pu, 1e-15, 1 - 1e-15))))
    out["auroc_at_iteration"] = auc
    if patience is not None:
        # what the authors' protocol reports: stop on the whole validation split, score it
        out["as_released_protocol"] = auc[str(kU)]["union"]
        # independent selection: stop on one half, score the other; both directions, averaged
        out["nested"] = dict(stop_on_A_score_B=auc[str(kA)]["B"], stop_on_B_score_A=auc[str(kB)]["A"])
        out["nested"]["mean"] = float(np.mean(list(out["nested"].values())))
        out["selection_effect"] = float(np.mean([auc[str(kU)]["B"] - auc[str(kA)]["B"],
                                                 auc[str(kU)]["A"] - auc[str(kB)]["A"]]))
        out["reproduction_check"] = dict(our_valid_logloss=float(cU[kU - 1]),
                                         released_valid_logloss=ORIGINAL_VALID_LOGLOSS,
                                         our_best_iteration=kU, released_best_iteration=10596)
    # GC-matched evaluation of the same fitted model (training untouched): the validation rows are
    # subsampled so that positives and negatives share one GC distribution
    gc = meta["gc"]
    k_eval = kU if patience is not None else ks[0]
    yu = np.r_[y[va_a], y[va_b]]
    pu = np.r_[pa[k_eval], pb[k_eval]]
    rows_u = np.r_[va_a, va_b]
    sub = pd.DataFrame({"Target": yu.astype(int)})
    keep = E.gc_match(sub, gc[rows_u], seed=SEED)
    out["gc_matched_eval"] = dict(n=int(len(keep)), auroc=float(roc_auc_score(yu[keep], pu[keep])),
                                  auroc_unmatched=float(roc_auc_score(yu, pu)),
                                  gc_only_unmatched=float(roc_auc_score(yu, gc[rows_u])),
                                  gc_only_matched=float(roc_auc_score(yu[keep], gc[rows_u][keep])))
    # The paper DECLARES that negatives match positives in number, length and strand (per protein).
    # Length is the declared property that is not delivered (single-feature AUROC 0.528, Table 7), so
    # restoring it within protein is a defect removal, not a change of target: equal numbers of
    # positives and negatives are kept in every (protein, length) cell of the validation rows.
    length = meta["length"].astype(np.int64)
    pid_u = meta["pid"][rows_u].astype(np.int64)
    cell = pid_u * 1000 + length[rows_u]
    rng = np.random.default_rng(SEED)
    keep_len = []
    for c in np.unique(cell):
        ip = np.where((cell == c) & (yu == 1))[0]
        ineg = np.where((cell == c) & (yu == 0))[0]
        m = min(len(ip), len(ineg))
        if m:
            keep_len.append(rng.choice(ip, m, replace=False))
            keep_len.append(rng.choice(ineg, m, replace=False))
    keep_len = np.sort(np.concatenate(keep_len))
    out["length_matched_eval"] = dict(
        n=int(len(keep_len)), auroc=float(roc_auc_score(yu[keep_len], pu[keep_len])),
        auroc_unmatched=float(roc_auc_score(yu, pu)),
        length_only_unmatched=float(roc_auc_score(yu, length[rows_u])),
        length_only_matched=float(roc_auc_score(yu[keep_len], length[rows_u][keep_len])))
    np.save(os.path.join(WORK, f"curves_{tag}.npy"), np.c_[cA, cB])
    _dump(out, args.out)


def _dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
    print(json.dumps(obj, indent=1)[:3000], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "scaler-check", "score-released", "retrain",
                                      "r1-settings", "sub400k"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="score-released: first N rows only")
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--leaf-trees", type=int, default=5)
    args = ap.parse_args()
    if args.stage == "features":
        stage_features(args)
    elif args.stage == "scaler-check":
        stage_scaler_check(args)
    elif args.stage == "score-released":
        stage_score_released(args)
    elif args.stage == "retrain":
        stage_retrain(args, "original")
    elif args.stage == "r1-settings":
        stage_retrain(args, "r1")
    elif args.stage == "sub400k":
        stage_retrain(args, "original", n_sub=400_000)


if __name__ == "__main__":
    main()
