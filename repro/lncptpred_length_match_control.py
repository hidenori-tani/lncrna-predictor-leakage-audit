"""Control for the length-match step of Section 3.5 (external review of R2, round 1).

Restoring the declared length match keeps a subset of the validation rows, so the change in score
could come from scoring fewer rows, or other proteins, rather than from the match. This scores the
model retrained with the original settings on the full benchmark (training untouched) on
  - random subsets with exactly the matched subset's number of positives and negatives of every
    protein, drawn without regard to length (200 draws), and
  - the matched subset itself, redrawn with 20 other seeds.
The predictions on the validation rows at the selected iteration are cached in lncptpred_full/
(git-ignored); the script first checks that it reproduces the retrain's recorded scores.

    cd repro && OMP_NUM_THREADS=1 nice -n 19 ./venv/bin/python lncptpred_length_match_control.py \
        --retrain ../revision_r2/results/lncptpred_retrain_original.json \
        --out ../revision_r2/results/lncptpred_length_match_control.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lncptpred_faithful as F  # noqa: E402

DRAWS = 200
REDRAWS = 20


def length_match(cell, yu, rng):
    """The retrain's matching rule: equal numbers of positives and negatives in every cell."""
    keep = []
    for c in np.unique(cell):
        ip = np.where((cell == c) & (yu == 1))[0]
        ineg = np.where((cell == c) & (yu == 0))[0]
        m = min(len(ip), len(ineg))
        if m:
            keep.append(rng.choice(ip, m, replace=False))
            keep.append(rng.choice(ineg, m, replace=False))
    return np.sort(np.concatenate(keep))


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rt = json.load(open(args.retrain))
    kU = rt["early_stopping"]["union"]["best"]
    P = F._paths()
    meta = np.load(P["meta"])
    y = meta["y"].astype(np.float64)
    tr, va_a, va_b = F.split_80_20(meta)
    rows_u = np.r_[va_a, va_b]
    yu = np.r_[y[va_a], y[va_b]]

    cache = os.path.join(F.WORK, f"valid_pred_original_k{kU}.npy")
    if os.path.exists(cache):
        pu = np.load(cache)
    else:
        X = np.load(P["X"], mmap_mode="r")
        mean, scale = F.fit_scaler(X, tr)
        print(f"scaler ({time.time()-t0:.0f}s)", flush=True)
        bst = lgb.Booster(model_file=os.path.join(F.WORK, "retrain_original.model.txt"))
        print(f"model loaded ({time.time()-t0:.0f}s)", flush=True)
        pu = np.r_[F.predict_rows(bst, X, va_a, mean, scale, num_iteration=kU),
                   F.predict_rows(bst, X, va_b, mean, scale, num_iteration=kU)]
        del bst
        np.save(cache, pu)
        print(f"predicted {len(pu):,} rows ({time.time()-t0:.0f}s)", flush=True)

    rec = rt["length_matched_eval"]
    full = float(roc_auc_score(yu, pu))
    length = meta["length"].astype(np.int64)[rows_u]
    pid = meta["pid"].astype(np.int64)[rows_u]
    cell = pid * 1000 + length                     # lengths are 10-202 nt: cells do not collide
    keep = length_match(cell, yu, np.random.default_rng(F.SEED))
    matched = float(roc_auc_score(yu[keep], pu[keep]))
    reproduced = dict(auroc_full=full, recorded_full=rec["auroc_unmatched"],
                      n_matched=int(len(keep)), recorded_n=rec["n"],
                      auroc_matched=matched, recorded_matched=rec["auroc"])
    ok = (abs(full - rec["auroc_unmatched"]) < 1e-9 and len(keep) == rec["n"]
          and abs(matched - rec["auroc"]) < 1e-9)
    reproduced["ok"] = bool(ok)
    print(reproduced, flush=True)
    if not ok:
        sys.exit("the cached predictions do not reproduce the retrain's recorded scores")

    # the control: per protein, as many positives and negatives as the matched subset keeps, at random
    proteins = np.unique(pid)
    m_p = {int(p): int(((pid[keep] == p) & (yu[keep] == 1)).sum()) for p in proteins}
    assert all(m_p[int(p)] == int(((pid[keep] == p) & (yu[keep] == 0)).sum()) for p in proteins)
    pos_of = {int(p): np.where((pid == p) & (yu == 1))[0] for p in proteins}
    neg_of = {int(p): np.where((pid == p) & (yu == 0))[0] for p in proteins}
    ctl, ctl_len = [], []
    for d in range(DRAWS):
        rng = np.random.default_rng(10_000 + d)
        sel = np.sort(np.concatenate(
            [rng.choice(pos_of[p], m, replace=False) for p, m in m_p.items() if m]
            + [rng.choice(neg_of[p], m, replace=False) for p, m in m_p.items() if m]))
        assert len(sel) == len(keep)
        ctl.append(float(roc_auc_score(yu[sel], pu[sel])))
        ctl_len.append(float(roc_auc_score(yu[sel], length[sel])))
    redraw, redraw_len = [], []
    for s in range(REDRAWS):
        k2 = length_match(cell, yu, np.random.default_rng(F.SEED + 1 + s))
        redraw.append(float(roc_auc_score(yu[k2], pu[k2])))
        redraw_len.append(float(roc_auc_score(yu[k2], length[k2])))
    ctl_a = np.array(ctl)
    out = dict(selected_iteration=kU, n_valid=int(len(yu)), reproduction=reproduced,
               matched=dict(n=int(len(keep)), auroc=matched,
                            length_only=float(roc_auc_score(yu[keep], length[keep]))),
               control_same_protein_counts=dict(
                   draws=DRAWS, seeds="10000 + draw", mean=float(ctl_a.mean()), sd=float(ctl_a.std(ddof=1)),
                   p2_5=float(np.percentile(ctl_a, 2.5)), p97_5=float(np.percentile(ctl_a, 97.5)),
                   min=float(ctl_a.min()), max=float(ctl_a.max()),
                   length_only_mean=float(np.mean(ctl_len)),
                   draws_at_or_below_matched=int((ctl_a <= matched).sum())),
               matched_redrawn=dict(seeds=f"{F.SEED + 1}..{F.SEED + REDRAWS}", auroc=redraw,
                                    min=float(min(redraw)), max=float(max(redraw)),
                                    length_only_max=float(max(redraw_len))),
               seconds=round(time.time() - t0, 1))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print({k: v for k, v in out.items() if k not in ("matched_redrawn",)}, flush=True)
    print("redrawn", out["matched_redrawn"]["min"], out["matched_redrawn"]["max"], flush=True)


if __name__ == "__main__":
    main()
