#!/usr/bin/env python
"""Every number the R2 text and tables quote, computed from the result files and nowhere else.

Writes
  key_numbers_r2.json   {"fmt": {key: string as printed}, "raw": {key: float}} -- fill_r2.py puts the
                        "fmt" strings into the text; check_numbers_r2.py traces the text back to them
  tables_r2.src.md      Tables 1-4 (static design tables) and 5-8 (generated from the results); Table 9
                        is a text block that fill_r2.py adds when it writes tables_r2.md
  supplementary_r2.md   Supplementary Tables S1-S5

Sources (all read-only):
  repro/dvmnet_revision_results.json, repro/dvmnet_sensitivity_results.json   R1 DVMnet runs
  results/dvmnet_r2.json          DVMnet rerun, training negatives from the training graph (R2)
  results/dvmnet_baselines.json   model families on the identical folds and negatives
  results/swap_draws_check.json   replay of every swap draw
  results/rnalight_baselines.json
  results/lncptpred_*.json, provenance/released_lncptpred_model_report.json
  ../revision/key_numbers.json    R1 numbers that R2 quotes unchanged (impossible pairs, n=400k runs)

--partial lets a missing source through (development only); the build refuses a partial file.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
REPRO = os.path.join(ROOT, "repro")

FMT, RAW, MISSING = {}, {}, []


def put(key, value, nd=3, signed=False, pct=False):
    if value is None:
        MISSING.append(key)
        return
    RAW[key] = float(value)
    if pct:
        FMT[key] = f"{value:.{nd}f}"
    elif signed:
        FMT[key] = f"{value:+.{nd}f}".replace("-", "−")
    else:
        FMT[key] = f"{value:.{nd}f}"


def put_text(key, text):
    FMT[key] = text


def load(path, required=True):
    if not os.path.exists(path):
        if required:
            MISSING.append(os.path.relpath(path, ROOT))
        return None
    with open(path) as fh:
        return json.load(fh)


def ci95(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def paired(a, b):
    """b - a over partitions; partition r of every condition uses cv seed 42 + 100 r."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    m, lo, hi = ci95(d)
    p = float(stats.ttest_rel(b, a).pvalue) if len(d) > 1 else float("nan")
    return dict(mean=m, lo=lo, hi=hi, p=p, n=len(d))


COND = {"rel": ("random", "verbatim"), "uni": ("random", "uniform"), "swap": ("random", "swap"),
        "cuni": ("cold", "uniform"), "cswap": ("cold", "swap"), "rdeg": ("random", "degmat"),
        "cdeg": ("cold", "degmat")}
FAMS = {"deg": "degree_sum", "pop": "popularity_lr", "nid": "node_id_lr", "comp": "composition_gbm",
        "d2v": "doc2vec_gbm", "seed": "seed_sites", "fam": "seed_family_lr", "dvm": "DVMnet"}
NODE_LEVEL = ("deg", "pop", "nid", "comp", "d2v")


def fam_value(summary, fam_key):
    """Training-graph-only scope where it exists (identical under random/uniform; the only valid one
    under every other non-released condition); DVMnet likewise prefers the R2 rerun."""
    name = FAMS[fam_key]
    for k in (name + "@trainonly", name):
        if k in summary:
            return summary[k]
    return None


def dvmnet_numbers(base, r1, r2):
    S = {c: base.get(sn) for c, sn in COND.items()}
    for c, s in S.items():
        if s is None:
            MISSING.append(f"baselines {c}")
            continue
        for f in FAMS:
            v = fam_value(s["summary"], f)
            if v is None:
                MISSING.append(f"dvm_{c}_{f}")
                continue
            put(f"dvm_{c}_{f}", v["mean"])
            put(f"dvm_{c}_{f}_sd", v["sd"])
    # DVMnet nested, its SD and the paired changes, from the DVMnet runs themselves
    dv = {}
    for c, sn in COND.items():
        # released and valid-pair uniform: identical under both training scopes, so the R1 runs stand;
        # every other condition must come from the R2 rerun -- never fall back to the defective R1 run
        src = r1.get(sn) if c in ("rel", "uni") else r2.get(sn)
        if src is not None and c not in ("rel", "uni") and src.get("train_scope") != "train":
            sys.exit(f"dvmnet_r2.json condition {sn} was not run with training negatives from the training graph")
        if src is None:
            MISSING.append(f"dvmnet run {c}")
            continue
        dv[c] = src
        put(f"dvm_{c}_dvm", src["summary"]["nested"]["mean"])
        put(f"dvm_{c}_dvm_sd", src["summary"]["nested"]["sd"])
        _, lo, hi = ci95([r["nested"] for r in src["per_repeat"]])
        put(f"dvm_{c}_dvm_lo", lo)
        put(f"dvm_{c}_dvm_hi", hi)
        put(f"dvm_{c}_final", src["summary"]["test_final"]["mean"])
        # the degree sum (training-graph degrees) on this condition's own test halves, from the run itself
        put(f"dvm_{c}_degtest", src["summary"]["degree_test"]["mean"])
        sb = [r["test_oracle"] - r["nested"] for r in src["per_repeat"]]
        m, lo, hi = ci95(sb)
        put(f"dvm_sb_{c}", m, nd=4, signed=True)
        put(f"dvm_sb_{c}_lo", lo, nd=4, signed=True)
        put(f"dvm_sb_{c}_hi", hi, nd=4, signed=True)
    if dv:
        sbs = {c: RAW.get(f"dvm_sb_{c}") for c in dv}
        put("dvm_sb_max", max(v for v in sbs.values() if v is not None), nd=3, signed=True)
    put("dvm_cdeg_train_final", dv["cdeg"]["summary"]["test_final"]["mean"] if "cdeg" in dv else None)

    def nested_vals(c):
        return [r["nested"] for r in dv[c]["per_repeat"]] if c in dv else None

    for tag, a, b in (("rel_uni", "rel", "uni"), ("uni_swap", "uni", "swap"), ("uni_cuni", "uni", "cuni"),
                      ("swap_cswap", "swap", "cswap"), ("cuni_cswap", "cuni", "cswap"), ("rel_swap", "rel", "swap")):
        va, vb = nested_vals(a), nested_vals(b)
        if va is None or vb is None:
            MISSING.append(f"dvm_d_{tag}")
            continue
        # paired by partition: the same partitions, in the same order, in both conditions -- a shorter
        # run is not paired on its common prefix (Codex review, 2026-10-22)
        ra = [r["repeat"] for r in dv[a]["per_repeat"]]
        rb = [r["repeat"] for r in dv[b]["per_repeat"]]
        if ra != rb:
            sys.exit(f"dvm_d_{tag}: conditions {a} and {b} do not have the same partitions ({ra} vs {rb})")
        d = paired(va, vb)
        put(f"dvm_d_{tag}", d["mean"], signed=True)
        put(f"dvm_d_{tag}_lo", d["lo"], signed=True)
        put(f"dvm_d_{tag}_hi", d["hi"], signed=True)
        RAW[f"dvm_d_{tag}_p"] = d["p"]
    # best single fold of the released protocol (the 0.87 comparison)
    if "rel" in dv:
        # the quantity the released script prints per fold: best epoch on the whole held-out fold
        folds = [f["full_best"] for r in dv["rel"]["per_repeat"] for f in r["folds"]]
        put("dvm_best_fold", max(folds), nd=4)
        put_text("dvm_n_folds", str(len(folds)))
    # ranges of the node-level families
    for c in ("rel", "uni", "swap", "cswap"):
        vals = [RAW[k] for k in (f"dvm_{c}_{f}" for f in NODE_LEVEL) if k in RAW]
        if vals:
            put(f"dvm_{c}_nodelevel_lo", min(vals))
            put(f"dvm_{c}_nodelevel_hi", max(vals))
    if "dvm_swap_nodelevel_lo" in FMT:
        put_text("dvm_swap_nodelevel_range", f"{FMT['dvm_swap_nodelevel_lo']}–{FMT['dvm_swap_nodelevel_hi']}")
    return dv


def artifact_numbers(base):
    """The R1 training-negative defect, shown on the node-identity model (R1 scope vs training graph)."""
    for c in ("rdeg", "cdeg", "cuni"):
        s = base.get(COND[c])
        if s is None:
            continue
        sm = s["summary"]
        if "node_id_lr" in sm:
            put(f"dvm_art_nid_{c}_global", sm["node_id_lr"]["mean"])
        if "node_id_lr@trainonly" in sm:
            put(f"dvm_art_nid_{c}_train", sm["node_id_lr@trainonly"]["mean"])


def r1_changed(r1all):
    """Values the previous version reported for the conditions whose training negatives it drew with
    global degrees (correction 1 of the response letter): quoted there as 'previous -> rerun'."""
    for c in ("rdeg", "cuni", "cdeg"):
        src = r1all.get(COND[c])
        if src is None:
            MISSING.append(f"R1 run {c}")
            continue
        put(f"dvm_r1_{c}_dvm", src["summary"]["nested"]["mean"], nd=4)
        put(f"dvm_r1_{c}_final", src["summary"]["test_final"]["mean"], nd=3)
        sb = [r["test_oracle"] - r["nested"] for r in src["per_repeat"]]
        put(f"dvm_r1_sb_{c}", float(np.mean(sb)), nd=4, signed=True)
    sbs = []
    for sn, src in r1all.items():
        if sn in (COND["rel"], COND["uni"], COND["rdeg"], COND["cuni"], COND["cdeg"]):
            sbs.append(float(np.mean([r["test_oracle"] - r["nested"] for r in src["per_repeat"]])))
    if sbs:
        put("dvm_r1_sb_max", max(sbs), nd=3, signed=True)


SENS = [("random", "alpha0.5"), ("random", "degmat"), ("random", "joint"), ("random", "nohub"),
        ("cold", "alpha0.5"), ("cold", "degmat"), ("cold", "joint"), ("cold", "nohub")]


def sensitivity_rows(base, r1all, r2):
    rows = []
    for sp, ng in SENS:
        new = r2.get((sp, ng))
        old = r1all.get((sp, ng))
        b = base.get((sp, ng))
        if new is None and old is None:
            continue
        if new is None:
            MISSING.append(f"R2 rerun of {sp}-{ng}")
        rows.append(dict(split=sp, neg=ng,
                         old=old["summary"]["nested"] if old else None,
                         new=new["summary"]["nested"] if new else None,
                         deg=(new or old)["summary"]["degree_test"],
                         nid_global=b["summary"].get("node_id_lr") if b else None,
                         nid_train=b["summary"].get("node_id_lr@trainonly") if b else None,
                         sb=(float(np.mean([p["test_oracle"] - p["nested"] for p in new["per_repeat"]]))
                             if new else None),
                         repeats=(new or old)["repeats"]))
    degs = [r["deg"]["mean"] for r in rows]
    if degs:
        put("dvm_sens_deg_lo", min(degs))
        put("dvm_sens_deg_hi", max(degs))
        put_text("dvm_sens_deg_range", f"{FMT['dvm_sens_deg_lo']}–{FMT['dvm_sens_deg_hi']}")
    news = [r["new"]["mean"] for r in rows if r["new"]]
    if news:
        put("dvm_sens_new_lo", min(news))
        put("dvm_sens_new_hi", max(news))
    for r in rows:
        tag = f"{'r' if r['split'] == 'random' else 'c'}{r['neg'].replace('.', '')}"
        if r["new"]:
            put(f"dvm_sens_{tag}_new", r["new"]["mean"])
            put(f"dvm_sens_{tag}_new_sd", r["new"]["sd"])
        if r["old"]:
            put(f"dvm_sens_{tag}_old", r["old"]["mean"], nd=4)
            put(f"dvm_sens_{tag}_old_sd", r["old"]["sd"], nd=4)
        # the other cells of Table S2, so that every number it prints traces to a key (the sweep passed
        # some of them only by coinciding with other values on the mock run, 2026-10-22)
        put(f"dvm_sens_{tag}_deg", r["deg"]["mean"])
        if r["nid_global"]:
            put(f"dvm_sens_{tag}_nidg", r["nid_global"]["mean"])
        if r["nid_train"]:
            put(f"dvm_sens_{tag}_nidt", r["nid_train"]["mean"])
        if r["sb"] is not None:
            put(f"dvm_sens_{tag}_sb", r["sb"], nd=4, signed=True)
    # the largest selection bias across every condition run (Table 6 and Supplementary Table S2):
    # dvmnet_numbers set it over the Table 6 conditions only
    every = [RAW[k] for k in RAW if re.fullmatch(r"dvm_sb_[a-z0-9]+", k) and k != "dvm_sb_max"]
    every += [r["sb"] for r in rows if r["sb"] is not None]
    if every:
        put("dvm_sb_max", max(every), nd=3, signed=True)
    # how far DVMnet rises above a model of node identity alone, trained on the same negatives, with the
    # published split (Section 3.3 and the Limitations)
    gaps = [r["new"]["mean"] - r["nid_train"]["mean"] for r in rows
            if r["split"] == "random" and r["new"] and r["nid_train"]]
    if gaps:
        put("dvm_sens_r_vs_nid_max", max(gaps))
    return rows


def rnalight_numbers(rl):
    fams = ["length_lr", "gc_lr", "length_gc_lr", "dinuc_lr", "kmer_lr", "kmer_knn", "kmer_lgbm"]
    short = {"length_lr": "length", "gc_lr": "gc", "length_gc_lr": "length_gc", "dinuc_lr": "dinuc",
             "kmer_lr": "kmer_lr", "kmer_knn": "kmer_knn", "kmer_lgbm": "kmer_lgbm"}
    for f in fams:
        s = short[f]
        put(f"rl_pub_{s}", rl["published"][f])
        put(f"rl_cv_{s}", rl["cv"][f]["mean"])
        put(f"rl_cv_{s}_sd", rl["cv"][f]["sd"])
        put(f"rl_matched_{s}", rl["matched"][f])
        put(f"rl_h05_{s}", rl["homology_05"][f]["mean"])
        put(f"rl_h08_{s}", rl["homology_08"][f]["mean"])
    full, di = rl["cv"]["kmer_lgbm"]["mean"], rl["cv"]["dinuc_lr"]["mean"]
    put("rl_cv_dinuc_share", 100 * (di - 0.5) / (full - 0.5), nd=0, pct=True)
    put_text("rl_matched_n", f"{rl['matched']['n']:,}")
    rs = rl["resplit"]["kmer_lgbm"]
    put("rl_resplit_mean", rs["mean"])
    put("rl_resplit_sd", rs["sd"])
    put("rl_resplit_p05", rs["p05"])
    put("rl_resplit_p95", rs["p95"])
    pct = rs["percentile_of_published"]
    RAW["rl_resplit_pct"] = pct
    FMT["rl_resplit_pct"] = f"{pct:.0f}th"
    # how far the published split's score sits above what a split drawn the same way gives on average
    put("rl_pub_vs_resplit", rs["published"] - rs["mean"])
    return fams


def lncptpred_numbers(rep, feats, scaler, srel, rt, r1full, o400):
    p = rep["classifier"]["params"]
    put_text("lnc_num_leaves", str(p["num_leaves"]))
    put_text("lnc_max_depth", str(p["max_depth"]))
    put("lnc_lr", p["learning_rate"], nd=4)
    put("lnc_l1", p["reg_alpha"], nd=2)
    put("lnc_l2", p["reg_lambda"], nd=2)
    put("lnc_colsample", p["colsample_bytree"], nd=3)
    put_text("lnc_max_rounds", f"{p['n_estimators']:,}")
    put_text("lnc_best_iter", f"{p['_best_iteration']:,}")
    ev = rep["evals_result_summary"]["valid_1/binary_logloss"]
    put_text("lnc_recorded_iter", f"{ev['n_iterations_recorded']:,}")
    put_text("lnc_n_train", f"{rep['scaler']['n_samples_seen_value']:,}")
    put("lnc_valid_logloss", ev["at_best_iteration"], nd=4)
    put("lnc_train_logloss", rep["evals_result_summary"]["training/binary_logloss"]["at_best_iteration"], nd=4)
    if scaler:
        put("lnc_scaler_max_sd", scaler["max_mean_diff_in_sd_released_order"], nd=4)
    if srel:
        put("lnc_rel_logloss_all", srel["logloss_all_rows"], nd=4)
        put("lnc_rel_logloss_expected", srel["expected_logloss_if_features_match"], nd=4)
        if "n_scored" in srel:
            put_text("lnc_rel_n_scored", f"{srel['n_scored']:,}")
        # leaf counts: the model stores how many TRAINING rows reach each leaf; ours count all rows, so a
        # leaf's ratio is 0.8 if our rows route like theirs (the overall ratio is 0.8 by construction and
        # says nothing; the per-leaf spread and the absence of leaves with more stored rows than ours do)
        lc = srel.get("leaf_count_checks") or []
        if lc:
            put_text("lnc_leaf_trees", str(len(lc)))
            put("lnc_leaf_ratio_median", float(np.median([t["ratio_median_leaves_ge1000"] for t in lc])), nd=3)
            put("lnc_leaf_ratio_lo", min(t["ratio_p05_p95_leaves_ge1000"][0] for t in lc), nd=2)
            put("lnc_leaf_ratio_hi", max(t["ratio_p05_p95_leaves_ge1000"][1] for t in lc), nd=2)
            put("lnc_leaf_exceed", sum(t["leaves_with_more_stored_than_ours"] for t in lc), nd=0)
    if rt:
        rc = rt["reproduction_check"]
        put("lnc_rt_logloss", rc["our_valid_logloss"], nd=4)
        put_text("lnc_rt_best_iter", f"{rc['our_best_iteration']:,}")
        put("lnc_rt_auc", rt["as_released_protocol"])
        put("lnc_rt_nested", rt["nested"]["mean"])
        put("lnc_rt_sel_effect", rt["selection_effect"], nd=5, signed=True)
        g = rt["gc_matched_eval"]
        put_text("lnc_rt_gc_n", f"{g['n']:,}")
        put("lnc_rt_gc_auc", g["auroc"])
        put("lnc_rt_gc_unmatched", g["auroc_unmatched"])
        put("lnc_rt_gc_only_matched", g["gc_only_matched"])
        put("lnc_rt_gc_only_unmatched", g["gc_only_unmatched"])
        ln = rt["length_matched_eval"]
        put_text("lnc_rt_len_n", f"{ln['n']:,}")
        put("lnc_rt_len_auc", ln["auroc"])
        put("lnc_rt_len_unmatched", ln["auroc_unmatched"])
        put("lnc_rt_len_only_matched", ln["length_only_matched"])
        put("lnc_rt_len_only_unmatched", ln["length_only_unmatched"])
        put("lnc_rt_d_len", ln["auroc"] - ln["auroc_unmatched"], signed=True)
        put("lnc_rt_d_len_abs", abs(ln["auroc"] - ln["auroc_unmatched"]))
        put("lnc_rt_d_gc", g["auroc"] - g["auroc_unmatched"], signed=True)
        put_text("lnc_rt_iterations", f"{rt['iterations_run']:,}")
        # a tolerance for the reproduction that the data supply: how far apart the two random halves of
        # our own validation split are at the selected iteration (the authors' split seed is unrecorded)
        cpath = os.path.join(REPRO, "lncptpred_full", "curves_original.npy")
        if os.path.exists(cpath):
            cab = np.load(cpath, allow_pickle=False)
            kU = rt["early_stopping"]["union"]["best"]
            a, b = float(cab[kU - 1, 0]), float(cab[kU - 1, 1])
            put("lnc_rt_logloss_a", a, nd=4)
            put("lnc_rt_logloss_b", b, nd=4)
            put("lnc_rt_half_diff", abs(a - b), nd=4)
            # how flat the curve is where stopping happens: the union log-loss (the size-weighted mean
            # of the two halves, as the run computes it) 1,000 iterations before the best one, minus
            # its value at the best one
            if kU > 1000:
                w = np.array([rt["n_A"], rt["n_B"]], dtype=float) / (rt["n_A"] + rt["n_B"])
                put("lnc_rt_tail_drop", float(cab[kU - 1001] @ w) - float(cab[kU - 1] @ w), nd=4)
            gap = rc["our_valid_logloss"] - rep["evals_result_summary"]["valid_1/binary_logloss"]["at_best_iteration"]
            put("lnc_rt_vs_authors", gap, nd=4, signed=True)
            put("lnc_rt_gap_abs", abs(gap), nd=4)
            put("lnc_rt_gap_ratio", abs(gap) / abs(a - b), nd=1)
            # how AUROC moves with log-loss near the selected iteration, from the iterations at which the run
            # recorded both (the best iterations of the two halves and of the union): the AUROC the authors'
            # log-loss corresponds to at that rate
            pts = sorted((v["union_logloss"], v["union"]) for v in rt["auroc_at_iteration"].values())
            if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
                slope = (pts[0][1] - pts[-1][1]) / (pts[-1][0] - pts[0][0])
                put("lnc_rt_auc_per_mll", slope * 0.001, nd=4)
                put("lnc_rt_auc_at_authors", rt["as_released_protocol"] + slope * gap)
                put("lnc_rt_auc_gain_at_authors", slope * gap)
        else:
            MISSING.append("repro/lncptpred_full/curves_original.npy")
    if r1full:
        put("lnc_r1full_auc", r1full["auroc_at_iteration"][str(r1full["iterations_run"])]["union"])
    if o400:
        put("lnc_o400_auc", o400["as_released_protocol"])
        put("lnc_o400_logloss", o400["reproduction_check"]["our_valid_logloss"], nd=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()

    r1_list = load(os.path.join(REPRO, "dvmnet_revision_results.json"))
    r1_sens = load(os.path.join(REPRO, "dvmnet_sensitivity_results.json"))
    r1 = {(c["split"], c["negatives"]): c for c in (r1_list or [])}
    r1all = dict(r1)
    r1all.update({(c["split"], c["negatives"]): c for c in (r1_sens or [])})
    r2_list = load(os.path.join(RES, "dvmnet_r2.json"))
    r2 = {(c["split"], c["negatives"]): c for c in (r2_list or [])}
    b = load(os.path.join(RES, "dvmnet_baselines.json"))
    base = {(c["split"], c["negatives"]): c for c in (b["results"] if b else [])}
    swap = load(os.path.join(RES, "swap_draws_check.json"))
    if swap:
        sel = swap["sets"]["cold/selection"]
        if sel["redrawn"] != sel.get("redrawn_minimum"):
            sys.exit("swap_draws_check.json: re-drawn pairs exceed the max-flow minimum (rerun verify_swap_draws.py)")
        put("swap_sel_redrawn_pct", sel["redrawn_pct"], nd=1, pct=True)
        put_text("swap_sel_redrawn_n", str(sel["redrawn"]))
        put_text("swap_sel_inexact", str(sel["draws"] - sel["draws_exact"]))
        put_text("swap_sel_draws", str(sel["draws"]))
        sh = sel.get("redrawn_draws_top_lncrna_share", [])
        if sh:
            put_text("swap_sel_top_share_range", f"{100 * min(sh):.0f}–{100 * max(sh):.0f}")
        put_text("swap_train_draws", f"{swap['sets']['random/train']['draws'] + swap['sets']['cold/train']['draws']:,}")
    dvmnet_numbers(base, r1, r2)
    r1_changed(r1all)
    artifact_numbers(base)
    sens_rows = sensitivity_rows(base, r1all, r2)
    # how often the released training sampler presents a held-out positive as a negative (Section 2.4)
    ho = load(os.path.join(RES, "released_heldout_negatives.json"))
    if ho:
        put("dvm_rel_heldout_pct", 100 * ho["heldout_hit_fraction"], nd=2, pct=True)
        put("dvm_rel_heldout_per_fold", ho["heldout_hits_per_fold_mean"], nd=1)
    rl = load(os.path.join(RES, "rnalight_baselines.json"))
    if rl:
        rnalight_numbers(rl)
    r1k = load(os.path.join(ROOT, "revision", "key_numbers.json"))
    if r1k:
        ex = r1k["lncptpred_exact"]
        put("lnc_r1_400k", ex["exact lncRNA 351 + protein 369"]["auroc_mean"])
        put("lnc_r1_400k_4", ex["exact lncRNA 351 + protein 369"]["auroc_mean"], nd=4)
        sc = r1k["derived"]["lncptpred_exact_scaling"]
        put("lnc_r1_6000", sc["6000"], nd=4)
        put("lnc_r1_20000", sc["20000"], nd=4)
    rep = load(os.path.join(HERE, "provenance", "released_lncptpred_model_report.json"))
    lncptpred_numbers(rep, load(os.path.join(RES, "lncptpred_features.json"), False),
                      load(os.path.join(RES, "lncptpred_scaler_check.json"), False),
                      load(os.path.join(RES, "lncptpred_score_released.json")),
                      load(os.path.join(RES, "lncptpred_retrain_original.json")),
                      load(os.path.join(RES, "lncptpred_r1_settings_full.json")),
                      load(os.path.join(RES, "lncptpred_original_n400k.json")))
    # the two-by-two of settings (earlier, recorded) by data (400,000 rows, all rows) behind the step
    # from the previous version's 0.79 (Section 3.1): each step read on its own, not as shares
    q = ("lnc_r1_400k", "lnc_r1full_auc", "lnc_o400_auc", "lnc_rt_auc")
    if all(k in RAW for k in q):
        r1_400, r1_all, rec_400, rec_all = (RAW[k] for k in q)
        put("lnc_d_set_400k", rec_400 - r1_400)
        put("lnc_d_set_full", rec_all - r1_all)
        put("lnc_d_data_r1", r1_all - r1_400)
        put("lnc_d_data_rec", rec_all - rec_400)
    # the control for the length-match step (Section 3.5): random subsets with the matched subset's
    # number of positives and negatives of every protein, drawn without regard to length
    lc = load(os.path.join(RES, "lncptpred_length_match_control.json"))
    if lc:
        c, rd = lc["control_same_protein_counts"], lc["matched_redrawn"]
        put_text("lnc_lenctl_draws", str(c["draws"]))
        put("lnc_lenctl_mean", c["mean"])
        put("lnc_lenctl_lo", c["p2_5"])
        put("lnc_lenctl_hi", c["p97_5"])
        put("lnc_lenctl_len_only", c["length_only_mean"])
        put_text("lnc_lenctl_redraws", str(len(rd["auroc"])))
        put("lnc_lenmatch_redraw_lo", rd["min"])
        put("lnc_lenmatch_redraw_hi", rd["max"])
        put("lnc_lenctl_d_abs", c["mean"] - lc["matched"]["auroc"])
        put("lnc_lenctl_d", lc["matched"]["auroc"] - c["mean"], signed=True)
    out = dict(fmt=FMT, raw=RAW, missing=sorted(set(MISSING)), partial=bool(MISSING))
    with open(os.path.join(HERE, "key_numbers_r2.json"), "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"key numbers: {len(FMT)}; missing: {len(set(MISSING))}")
    if MISSING:
        print("  missing:", ", ".join(sorted(set(MISSING))[:40]))
        if not args.partial:
            sys.exit(1)
    import make_tables_r2
    make_tables_r2.write(FMT, RAW, base, r1all, r2, rl, sens_rows)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
