#!/usr/bin/env python
"""Regenerate the tables of the SAGMB.2026.0060 revision from the results JSON files.

The manuscript states that "every number in this paper is produced by a script in the released
repository; no value reported here is transcribed by hand from a log".  The revision adds three
new experiments, so the tables that carry their numbers are generated here rather than edited.

Inputs (written by the experiment scripts, one JSON each)
  repro/dvmnet_revision_results.json      repeated CV + nested checkpoint selection  (R2-2, R2-3)
  repro/dvmnet_sensitivity_results.json   negative-sampling sensitivity              (R2-1)
  repro/lncptpred_exact_results.json      features rebuilt from the released code    (R2-4)
  repro/dvmnet_model_results.json         the submitted run, kept for Table 3 and comparison
  repro/lncptpred_model_results.json      the submitted LncPTPred run

Outputs
  revision/tables_r1.md      Tables 1-9
  revision/key_numbers.json  every value the prose is allowed to quote, so check_numbers.py can
                             verify the text against the runs rather than against my memory
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent / "repro"
OUT_MD = HERE / "tables_r1.md"
OUT_KEY = HERE / "key_numbers.json"

# condition -> how the paper names it
NAME = {"random-verbatim": "verbatim (as released)", "random-uniform": "random-uniform",
        "random-degmat": "random-degmat", "cold-uniform": "cold-uniform",
        "cold-degmat": "cold-degmat"}
ORDER = ["random-verbatim", "random-uniform", "random-degmat", "cold-uniform", "cold-degmat"]


def load(name, required=True):
    p = REPRO / name
    if not p.exists():
        if required:
            sys.exit(f"まだ結果が無い: {p}")
        return None
    return json.load(open(p))


def ci95(vals):
    """t-based 95% CI over independent CV partitions (repeats), not over folds."""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan"), float("nan")
    se = v.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def paired(a, b):
    """Paired over repeats: repeat r of every condition uses the same CV partition."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m, lo, hi = ci95(d)
    dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
    t, p = stats.ttest_rel(a, b)
    return dict(mean=m, lo=lo, hi=hi, dz=dz, t=float(t), p=float(p), n=len(d))


def f(x, nd=4):
    return f"{x:.{nd}f}"


def signed(x, nd=4):
    return f"{x:+.{nd}f}"


def main():
    rev = {r["condition"]: r for r in load("dvmnet_revision_results.json")}
    sens = load("dvmnet_sensitivity_results.json", required=False)
    sens = {r["condition"]: r for r in sens} if sens else {}
    lex = load("lncptpred_exact_results.json", required=False)
    lex_scale = {n: load(f"lncptpred_exact_n{n}.json", required=False) for n in (6000, 20000)}
    old_dvm = {r["condition"]: r for r in load("dvmnet_model_results.json")}

    missing = [c for c in ORDER if c not in rev]
    if missing:
        sys.exit(f"Stage A が未完了（欠けている条件: {missing}）")

    key = {"dvmnet": {}, "sensitivity": {}, "lncptpred_exact": {}}
    for c in ORDER:
        s = rev[c]["summary"]
        key["dvmnet"][c] = {k: {kk: s[k][kk] for kk in ("mean", "sd", "values")}
                            for k in ("nested", "test_oracle", "test_final", "full_best",
                                      "degree_test", "selection_bias")}
        key["dvmnet"][c]["repeats"] = rev[c]["repeats"]

    def vals(c, k):
        return rev[c]["summary"][k]["values"]

    # ---------------------------------------------------------------- Table 4
    t4 = ["Table 4. Protocol ablation of DVMnet under repeated cross-validation. "
          f"Each condition is run over {rev[ORDER[0]]['repeats']} independent 5-fold partitions"
          ". Values are the mean ± SD over partitions. \"nested\" selects the "
          "epoch on a held-out selection split and scores the disjoint test split; "
          "\"oracle\" selects on the test split itself and is therefore an upper bound; "
          "\"whole fold, best epoch\" is the quantity an earlier version of this analysis "
          "reported. \"degree\" is "
          "a zero-parameter degree-sum heuristic on the identical folds and negatives.", "",
          "| condition | nested | oracle | whole fold, best epoch | degree |",
          "|---|---|---|---|---|"]
    for c in ORDER:
        s = rev[c]["summary"]
        row = (f"| {NAME[c]} | {f(s['nested']['mean'])} ± {f(s['nested']['sd'])} "
               f"| {f(s['test_oracle']['mean'])} ± {f(s['test_oracle']['sd'])} "
               f"| {f(s['full_best']['mean'])} ± {f(s['full_best']['sd'])} "
               f"| {f(s['degree_test']['mean'])} |")
        if c == "random-degmat":
            row = row.replace(f"| {f(s['nested']['mean'])} ±", f"| **{f(s['nested']['mean'])}** ±", 1)
        t4.append(row)

    # ---------------------------------------------------------------- Table 5
    comps = [
        ("**degree-matched negatives** (split unchanged)", "random-degmat", "random-uniform"),
        ("entity-disjoint split (negatives uniform)", "cold-uniform", "random-uniform"),
        ("both controls", "cold-degmat", "random-uniform"),
        ("restricting negatives to valid pair types", "random-uniform", "random-verbatim"),
    ]
    t5 = ["Table 5. Paired differences in the nested estimate, over the independent partitions of "
          "Table 4. Pairing is at the level of the partition: partition *r* of every condition uses "
          "the same split, so the differences are paired and the partitions, unlike the folds "
          "within one partition, do not share training sets. What these intervals describe is the "
          "stability of the estimate when the partition is redrawn, not sampling error over "
          "datasets: every partition reuses the same 1,057 released edges, so the intervals "
          "are narrow by construction, and the effect sizes are statements about "
          "reproducibility under resampling rather than about a population of benchmarks. "
          "The CI carries the claim; "
          "p-values are given for completeness only.", "",
          "| comparison (nested) | mean diff | 95% CI | *d*z | t | p |", "|---|---|---|---|---|---|"]
    key["paired"] = {}
    for label, a, b in comps:
        r = paired(vals(a, "nested"), vals(b, "nested"))
        key["paired"][label] = r
        t5.append(f"| {label} | {signed(r['mean'])} | [{signed(r['lo'])}, {signed(r['hi'])}] "
                  f"| {r['dz']:.2f} | {r['t']:.1f} | {r['p']:.1e} |")

    # ---------------------------------------------------------------- Table 6
    t6 = ["Table 6. Checkpoint selection, measured against an independent reference. "
          "The epoch is chosen on a selection split that is disjoint from the test split "
          "\u2014 for entity-disjoint conditions the two halves hold out different lncRNAs \u2014 "
          "and the bias is the amount recovered by selecting on the test split instead. An earlier "
          "version of this analysis estimated it as (best epoch \u2212 final epoch) on the "
          "evaluation fold, which is reproduced in the last column and is a different quantity.", "",
          "| condition | nested | oracle | selection bias | 95% CI | naive (best \u2212 final) |",
          "|---|---|---|---|---|---|"]
    for c in ORDER:
        s = rev[c]["summary"]
        m, lo, hi = ci95(s["selection_bias"]["values"])
        oc = old_dvm.get("verbatim" if c == "random-verbatim" else c)
        old_gap = (oc["best_mean"] - oc["final_mean"]) if oc else float("nan")
        t6.append(f"| {NAME[c]} | {f(s['nested']['mean'])} | {f(s['test_oracle']['mean'])} "
                  f"| {signed(m)} | [{signed(lo)}, {signed(hi)}] | {signed(old_gap, 3)} |")
        key["dvmnet"][c]["selection_bias_ci"] = [m, lo, hi]
        key["dvmnet"][c]["submitted_best_minus_final"] = old_gap

    # ---------------------------------------------------------------- Table 8
    t8 = []
    if sens:
        t8 = ["Table 8. Sensitivity of the DVMnet result to how negatives are constructed. "
              "`alpha` draws each endpoint with probability proportional to "
              "its degree raised to the power alpha, so alpha = 0 is the uniform scheme and "
              "alpha = 1 is the degree-matched scheme of Table 4; `joint` matches the binned joint "
              "endpoint-degree distribution of the fold's positives rather than the two marginals; "
              "`nohub` is the alpha = 1 scheme with hub-hub pairs excluded, removing exactly the "
              "population most likely to contain unobserved true interactions. Values are the "
              "nested estimate, mean ± SD over independent partitions.", "",
              "| split | negative scheme | nested | degree |", "|---|---|---|---|"]
        # explicit order: the alpha dose-response first, then the two variants, so the
        # monotone trend in alpha is readable off the table rather than sorted away.
        NEG_LABEL = {"uniform": "alpha = 0 (uniform)", "alpha0.5": "alpha = 0.5",
                     "degmat": "alpha = 1 (degree-matched)", "joint": "joint degree match",
                     "nohub": "alpha = 1, hub-hub pairs excluded"}
        NEG_ORDER = ["uniform", "alpha0.5", "degmat", "joint", "nohub"]
        pool = {}
        for c in ORDER:
            if c != "random-verbatim":
                pool[c] = rev[c]["summary"]
        for c, r in sens.items():
            pool[c] = r["summary"]
            key["sensitivity"][c] = {k: r["summary"][k] for k in ("nested", "degree_test")}
        for split in ("random", "cold"):
            for neg in NEG_ORDER:
                sm = pool.get(f"{split}-{neg}")
                if sm is None:
                    continue
                t8.append(f"| {split} | {NEG_LABEL[neg]} "
                          f"| {f(sm['nested']['mean'])} ± {f(sm['nested']['sd'])} "
                          f"| {f(sm['degree_test']['mean'])} |")

    # ---------------------------------------------------------------- Table 9
    t9 = []
    if lex:
        t9 = ["Table 9. LncPTPred with the feature construction rebuilt exactly from the authors' "
              "released code. AUROC is the mean ± SD over five stratified "
              "folds at n = 400,000, with LightGBM settings identical across rows.", "",
              "| feature set | features | AUROC |", "|---|---|---|"]
        LEX_LABEL = {
            "exact lncRNA 351 only": "lncRNA side only (the released 351)",
            "exact lncRNA 351 + protein 369":
                "the released feature set (351 lncRNA + 369 protein)",
            "exact lncRNA 351 + protein categorical":
                "351 lncRNA + protein identity (a superset of any fixed protein lookup)",
            "released minus structure/strand (k-mer fractions + protein 369)":
                "released set minus the structure block and Strand",
            "released minus strand (351-1 + protein 369)": "released set minus Strand",
            "exact full [GC-matched]":
                "the released feature set, evaluated on a GC-matched subsample",
        }
        for r in lex:
            t9.append(f"| {LEX_LABEL.get(r['condition'], r['condition'])} | {r['n_features']} "
                      f"| {f(r['auroc_mean'])} ± {f(r['auroc_std'])} |")
            key["lncptpred_exact"][r["condition"]] = {"auroc_mean": r["auroc_mean"],
                                                      "auroc_std": r["auroc_std"],
                                                      "n_features": r["n_features"],
                                                      "n": r["n"]}

    # ---------------------------------------------------- derived values the prose quotes
    # Anything the text is allowed to state must be computed here, so check_numbers.py can
    # trace it back to a run instead of to a transcription.
    der = {}
    allf = [fd["full_best"] for rp in rev["random-verbatim"]["per_repeat"] for fd in rp["folds"]]
    der["best_single_fold_verbatim"] = max(allf)
    der["n_folds_total"] = len(allf)
    d = np.asarray(vals("random-degmat", "nested")) - np.asarray(vals("random-uniform", "nested"))
    der["per_partition_headline_diff"] = [float(x) for x in d]
    der["max_deviation_from_mean"] = float(np.max(np.abs(d - d.mean())))
    der["degree_margin"] = {}
    for c in ORDER:
        a = np.asarray(vals(c, "nested")); b = np.asarray(rev[c]["summary"]["degree_test"]["values"])
        tt, pp = stats.ttest_rel(a, b)
        der["degree_margin"][c] = {"mean": float((a - b).mean()), "p": float(pp)}
    worst = max(((fd["test_oracle"] - fd["nested"], c, rp["repeat"])
                 for c in ORDER for rp in rev[c]["per_repeat"] for fd in rp["folds"]),
                key=lambda x: x[0])
    der["max_single_fold_selection_bias"] = {"value": worst[0], "condition": worst[1],
                                             "partition": worst[2]}
    # --- negative-sampling sensitivity: how much of the fall is the construction itself?
    # The hub-excluded sampler removes exactly the (hub, hub) non-edges the positive-unlabelled
    # objection identifies, so (nohub - degmat) is a direct measurement of that artefact, and its
    # share of the (uniform - degmat) spread says how much of the fall it can account for.
    if sens:
        sp = {}
        for split in ("random", "cold"):
            u = rev.get(f"{split}-uniform")
            g = rev.get(f"{split}-degmat")
            nh = sens.get(f"{split}-nohub")
            jt = sens.get(f"{split}-joint")
            if not (u and g and nh):
                continue
            um = u["summary"]["nested"]["mean"]
            gm = g["summary"]["nested"]["mean"]
            nm = nh["summary"]["nested"]["mean"]
            e = dict(uniform=um, degmat=gm, nohub=nm,
                     spread=um - gm, pu_artefact=nm - gm,
                     pu_share=(nm - gm) / (um - gm))
            if jt:
                e["joint"] = jt["summary"]["nested"]["mean"]
                e["joint_drop"] = jt["summary"]["nested"]["mean"] - um
            sp[split] = e
        der["sensitivity_spread"] = sp
        # the widest and narrowest nested value seen anywhere in the family, per split
        der["sensitivity_range"] = {}
        for split in ("random", "cold"):
            vs = [r["summary"]["nested"]["mean"] for c, r in
                  list(rev.items()) + list(sens.items())
                  if c.startswith(split + "-") and c != "random-verbatim"]
            if vs:
                der["sensitivity_range"][split] = {"min": min(vs), "max": max(vs), "n": len(vs)}

    # --- LncPTPred, rebuilt exactly from the released code: the contrasts the prose quotes
    if lex:
        L = {r["condition"]: r["auroc_mean"] for r in lex}
        full = L.get("exact lncRNA 351 + protein 369")
        lnc = L.get("exact lncRNA 351 only")
        if full and lnc:
            der["lncptpred_exact"] = {
                "protein_gain": full - lnc,
                "protein_identity_gain": L["exact lncRNA 351 + protein categorical"] - lnc,
                "struct_strand_cost":
                    full - L["released minus structure/strand (k-mer fractions + protein 369)"],
                "strand_cost": full - L["released minus strand (351-1 + protein 369)"],
                "gc_cost": full - L["exact full [GC-matched]"],
                "lnc_share_above_chance": (lnc - 0.5) / (full - 0.5),
            }
        sc = {}
        for n, rows in lex_scale.items():
            if rows:
                m = {r["condition"]: r["auroc_mean"] for r in rows}
                sc[str(n)] = m.get("exact lncRNA 351 + protein 369")
        if sc:
            sc[str(lex[0]["n"])] = full
            der["lncptpred_exact_scaling"] = sc

    if "random-degmat" in old_dvm:
        der["submitted_degree_baseline_random_degmat"] = old_dvm["random-degmat"]["degree_baseline_mean"]
    key["derived"] = der

    # ------------------------------------------------------------ assemble
    old = OUT_MD.read_text(encoding="utf-8")
    def block(n):
        i = old.index(f"Table {n}.")
        j = old.index(f"Table {n+1}.") if f"Table {n+1}." in old else len(old)
        return old[i:j].rstrip()

    parts = ["# Tables", "", "What lncRNA predictors measure — Hidenori Tani", "",
             block(1), "", block(2), "", block(3), "",
             "\n".join(t4), "", "\n".join(t5), "", "\n".join(t6), "", block(7).rstrip()]
    if t8:
        parts += ["", "\n".join(t8)]
    if t9:
        parts += ["", "\n".join(t9)]
    OUT_MD.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    OUT_KEY.write_text(json.dumps(key, indent=2), encoding="utf-8")

    print(f"  ✔ {OUT_MD.name}  （表 {7 + bool(t8) + bool(t9)} 点）")
    print(f"  ✔ {OUT_KEY.name}")
    print("\n=== 主要な数値 ===")
    for c in ORDER:
        s = rev[c]["summary"]
        print(f"  {NAME[c]:<24} nested {f(s['nested']['mean'])} ± {f(s['nested']['sd'])}"
              f"   選択バイアス {signed(s['selection_bias']['mean'])}"
              f"   whole-fold best {f(s['full_best']['mean'])}")


if __name__ == "__main__":
    main()
