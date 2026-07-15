#!/usr/bin/env python
"""Formal paired tests over the DVMnet protocol-ablation folds.

Added 2026-07-16 after both external reviewers (gpt-5.1, gemini-2.5-pro) independently asked for the
"collapse" claim to be formalised rather than rest on non-overlapping SDs.

The five conditions share fold indices (same KFold seed), so the comparisons are PAIRED. n=5, so the
Wilcoxon signed-rank test has a floor of p=0.0625 (2^-4) for a two-sided test — it literally cannot
produce p<0.05 at this n. We therefore report the paired t-test as the primary test, Wilcoxon as a
distribution-free check, and — most usefully at n=5 — the paired mean difference with its CI and
Cohen's dz. Reporting the Wilcoxon floor explicitly is the honest move: it stops a reader from reading
"p=0.0625, n.s." as evidence of no effect.
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

def results_path(name):
    """Locate a results JSON next to this script, or in ../results.

    The working tree keeps scripts and results in one directory; the public release
    splits them into code/ and results/. This resolves both so the released file is
    byte-identical to the one that produced the published numbers.
    """
    for cand in (os.path.join(os.environ.get("AUDIT_RESULTS", HERE), name),
                 os.path.join(HERE, "..", "results", name)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"{name} not found next to the script or in ../results")


RESULTS = results_path("dvmnet_model_results.json")


def load():
    """Read per-fold AUROCs from the results JSON — the direct output of exp_dvmnet_model.py.

    This used to regex the run log instead, which was wrong twice over. The log prints
    4 decimal places, so every CI and dz was computed from rounded values; and it made a
    published number depend on parsing a human-readable log, which is exactly the kind of
    provenance we criticise the audited papers for. The JSON carries full precision and is
    a single script's output.
    """
    d = json.load(open(RESULTS))
    return {e["condition"]: {"best": np.array(e["best_per_fold"]),
                             "final": np.array(e["final_per_fold"]),
                             "degree": np.array(e["degree_per_fold"])}
            for e in d}


def paired(a, b, label):
    d = a - b
    n = len(d)
    t, pt = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = np.nan, np.nan
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.inf
    se = d.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    print(f"  {label}")
    print(f"    paired mean diff {d.mean():+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  Cohen's dz {dz:+.2f}")
    print(f"    paired t({n-1}) = {t:+.3f}, p = {pt:.2e}   |   Wilcoxon p = {pw:.4f}"
          f"{'  (= the n=5 two-sided floor, 2^-4)' if abs(pw - 0.0625) < 1e-9 else ''}")
    return dict(mean_diff=float(d.mean()), ci_low=float(ci[0]), ci_high=float(ci[1]),
                dz=float(dz), t=float(t), p_t=float(pt), p_wilcoxon=float(pw), n=int(n))


def main():
    f = load()
    for c, d in f.items():
        assert len(d["best"]) == 5, f"{c}: expected 5 folds, got {len(d['best'])}"
    print(f"loaded {len(f)} conditions x 5 folds from {os.path.basename(RESULTS)} (full precision)\n")

    out = {}
    print("=== headline: fixing ONLY the negatives (split unchanged) ===")
    out["negatives_effect_best"] = paired(
        f["random-degmat"]["best"], f["random-uniform"]["best"],
        "random-degmat vs random-uniform (best epoch)")

    print("\n=== the other ablation axes (best epoch) ===")
    out["pairtype_effect_best"] = paired(
        f["random-uniform"]["best"], f["verbatim"]["best"],
        "random-uniform vs verbatim  (restricting negatives to valid pair types)")
    out["split_effect_best"] = paired(
        f["cold-uniform"]["best"], f["random-uniform"]["best"],
        "cold-uniform vs random-uniform  (entity-disjoint split)")
    out["joint_effect_best"] = paired(
        f["cold-degmat"]["best"], f["random-uniform"]["best"],
        "cold-degmat vs random-uniform  (both controls)")

    print("\n=== epoch selection (best vs final, within condition) ===")
    for c in ["verbatim", "random-uniform", "random-degmat", "cold-uniform", "cold-degmat"]:
        out[f"epoch_gap_{c}"] = paired(f[c]["best"], f[c]["final"], f"{c}: best vs final")

    print("\n=== deep model vs zero-parameter degree baseline, per condition (best epoch) ===")
    for c in ["verbatim", "random-uniform", "random-degmat", "cold-uniform", "cold-degmat"]:
        out[f"vs_degree_{c}"] = paired(f[c]["best"], f[c]["degree"], f"{c}: deep model vs degree-sum")

    p = os.path.join(HERE, "dvmnet_stats.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
