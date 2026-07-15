#!/usr/bin/env python3
"""Check every DVMnet number asserted in the manuscript against the artefact JSONs.

Not a style check — an assertion that the prose agrees with the files we ship. Each entry
is (where_it_appears, claimed_value, computed_value). Anything that disagrees is printed
and the script exits non-zero.

This exists because on 2026-07-16 the draft carried three values (0.5064, 0.5803±0.0869,
0.3925) inherited from a superseded run, and I would not have caught them by reading.
"""
import json
import os
import pathlib
import sys

HERE = str(pathlib.Path(__file__).parent)

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


res = {e["condition"]: e for e in json.load(open(results_path("dvmnet_model_results.json")))}
st = json.load(open(results_path("dvmnet_stats.json")))

checks = []


def chk(where, claimed, computed, tol=5e-5):
    ok = abs(claimed - computed) <= tol
    checks.append((ok, where, claimed, computed))


# --- Table 1: the ablation ladder ---
for cond, best, bsd, fin, fsd, deg in [
    ("verbatim", 0.8234, 0.0230, 0.8021, 0.0364, 0.7600),
    ("random-uniform", 0.8028, 0.0168, 0.7375, 0.0181, 0.7502),
    ("random-degmat", 0.5098, 0.0287, 0.5065, 0.0283, 0.4581),
    ("cold-uniform", 0.6839, 0.0217, 0.5803, 0.0870, 0.4943),
    ("cold-degmat", 0.5735, 0.0279, 0.3924, 0.0217, 0.3841),
]:
    e = res[cond]
    chk(f"T1 {cond} best", best, round(e["best_mean"], 4))
    chk(f"T1 {cond} best SD", bsd, round(e["best_std"], 4))
    chk(f"T1 {cond} final", fin, round(e["final_mean"], 4))
    chk(f"T1 {cond} final SD", fsd, round(e["final_std"], 4))
    chk(f"T1 {cond} degree", deg, round(e["degree_baseline_mean"], 4))

# --- Table 2: paired effects ---
for where, key, md, lo, hi, dz in [
    ("T2 degree-matched", "negatives_effect_best", -0.2929, -0.3488, -0.2371, -6.51),
    ("T2 entity-disjoint", "split_effect_best", -0.1189, -0.1500, -0.0877, -4.74),
    ("T2 both controls", "joint_effect_best", -0.2293, -0.2886, -0.1699, -4.79),
    ("T2 pair-type", "pairtype_effect_best", -0.0207, -0.0595, +0.0182, -0.66),
]:
    s = st[key]
    chk(f"{where} mean", md, round(s["mean_diff"], 4))
    chk(f"{where} CI lo", lo, round(s["ci_low"], 4))
    chk(f"{where} CI hi", hi, round(s["ci_high"], 4))
    chk(f"{where} dz", dz, round(s["dz"], 2), tol=5e-3)

# --- epoch-selection table ---
for cond, gap, lo, hi in [
    ("verbatim", 0.021, -0.008, 0.051),
    ("random-uniform", 0.065, 0.045, 0.085),
    ("random-degmat", 0.003, -0.002, 0.009),
    ("cold-uniform", 0.104, -0.029, 0.236),
    ("cold-degmat", 0.181, 0.115, 0.247),
]:
    s = st[f"epoch_gap_{cond}"]
    chk(f"epoch {cond} gap", gap, round(s["mean_diff"], 3), tol=5e-4)
    chk(f"epoch {cond} CI lo", lo, round(s["ci_low"], 3), tol=5e-4)
    chk(f"epoch {cond} CI hi", hi, round(s["ci_high"], 3), tol=5e-4)

# --- prose claims ---
u = res["random-uniform"]["best_per_fold"]
g = res["random-degmat"]["best_per_fold"]
for i, claimed in enumerate([-0.305, -0.222, -0.304, -0.346, -0.289]):
    chk(f"prose per-fold delta {i}", claimed, round(g[i] - u[i], 3), tol=5e-4)

chk("prose impossible frac", 0.555, round(res["verbatim"]["impossible_pair_type_frac"], 3))
chk("prose headline from", 0.8028, round(res["random-uniform"]["best_mean"], 4))
chk("prose headline to", 0.5098, round(res["random-degmat"]["best_mean"], 4))
chk("prose max controlled", 0.5735, round(res["cold-degmat"]["best_mean"], 4))
chk("prose verbatim fold0", 0.8671, round(res["verbatim"]["best_per_fold"][0], 4))

cu = res["cold-uniform"]
chk("prose cu fold1 final", 0.4300, round(cu["final_per_fold"][1], 4))
chk("prose cu fold1 best", 0.6803, round(cu["best_per_fold"][1], 4))

chk("prose deep>degree degmat", 0.052, round(st["vs_degree_random-degmat"]["mean_diff"], 3), tol=5e-4)
chk("prose deep>degree colddeg", 0.189, round(st["vs_degree_cold-degmat"]["mean_diff"], 3), tol=5e-4)

# 'every one of five folds' — the collapse must hold fold-wise, not just on average
assert all(b < a for a, b in zip(u, g)), "claim 'in all five folds' is FALSE"

bad = [c for c in checks if not c[0]]
for ok, where, claimed, computed in checks:
    if not ok:
        print(f"  MISMATCH {where:28s} draft={claimed}  file={computed}")
print(f"\n{len(checks) - len(bad)}/{len(checks)} numeric claims verified against artefacts")
if bad:
    print(f"{len(bad)} MISMATCHES — draft must not be submitted in this state")
sys.exit(1 if bad else 0)
