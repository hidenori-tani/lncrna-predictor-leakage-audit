#!/usr/bin/env python
"""Figure 2 (R2): every model family on the RNAlight benchmark, under each protocol, by question.

(A) AUROC of seven model families under the published split and cross-validation (Q1), on the
length- and GC-matched subset (Q2) and under homology-aware splits at 50% and 80% identity (Q3).
(B) Where the published single split falls among 200 random splits of the same size (Q4).

Provenance: revision_r2/results/rnalight_baselines.json.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figcommon_r2 import JOURNAL, WIDTH_MM, layout_for_scholarone, save_tif  # noqa: E402
from figstyle import OKABE_ITO, mm, pt_scale, savefig, use_journal_style  # noqa: E402

HEIGHT_MM = 78.0
PROTOCOLS = [("published", "published\nsplit", "Q1"), ("cv", "cross-\nvalidation", "Q1"),
             ("matched", "length, GC\nmatched", "Q2"), ("homology_05", "homology\n50%", "Q3"),
             ("homology_08", "homology\n80%", "Q3")]
FAMILIES = [("length_lr", "length", "#4D4D4D", "o", False), ("gc_lr", "GC", "#4D4D4D", "s", False),
            ("length_gc_lr", "length + GC", "#4D4D4D", "^", True),
            ("dinuc_lr", "dinucleotide", "#8C8C8C", "D", True),
            ("kmer_lr", "k-mer, logistic", OKABE_ITO["blue"], "s", False),
            ("kmer_knn", "k-mer, nearest-neighbor", OKABE_ITO["blue"], "v", True),
            ("kmer_lgbm", "RNAlight", OKABE_ITO["vermillion"], "o", True)]


def val(rl, proto, key):
    v = rl[proto][key]
    if isinstance(v, dict):
        return v["mean"], v.get("sd", 0.0)
    return float(v), 0.0


IN = os.path.join(HERE, "results", "rnalight_baselines.json")
OUT = os.path.join(HERE, "figures", "Figure_2")


def figure(inp=IN):
    """Draw and return the figure; saving is main()'s (check_axes_clipping.py measures the return)."""
    rl = json.load(open(inp))

    use_journal_style(JOURNAL)
    pts = pt_scale(JOURNAL)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(mm(WIDTH_MM), mm(HEIGHT_MM)),
                                 gridspec_kw=dict(width_ratios=[2.3, 1.0]))
    rect = layout_for_scholarone(fig)

    offsets = np.linspace(-0.33, 0.33, len(FAMILIES))
    for pi, (proto, _lab, _q) in enumerate(PROTOCOLS):
        for fi, (key, _l, c, mk, filled) in enumerate(FAMILIES):
            m, sd = val(rl, proto, key)
            ax.errorbar(pi + offsets[fi], m, yerr=sd, fmt=mk, ms=4.6 if key == "kmer_lgbm" else 3.4, color=c,
                        mfc=c if filled else "white", mec=c, mew=0.8, elinewidth=0.6, capsize=0)
    ax.axhline(0.5, color="#999999", lw=0.6, ls=":")
    for pi in range(1, len(PROTOCOLS)):
        ax.axvline(pi - 0.5, color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_xlim(-0.55, len(PROTOCOLS) - 0.45)
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("AUROC", fontsize=pts["label"])
    ax.set_xticks(range(len(PROTOCOLS)))
    ax.set_xticklabels([p[1] for p in PROTOCOLS], fontsize=pts["small"])
    for pi, (_p, _l, q) in enumerate(PROTOCOLS):
        ax.text(pi, 0.842, q, ha="center", va="top", fontsize=pts["body"], color="#333333")
    handles = [plt.Line2D([], [], ls="none", marker=mk, ms=4.6 if k == "kmer_lgbm" else 3.4, color=c,
                          mfc=c if fl else "white", mec=c, mew=0.8, label=lab)
               for k, lab, c, mk, fl in FAMILIES]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              fontsize=pts["small"], handletextpad=0.3, columnspacing=1.0)
    ax.text(-0.1, 1.06, "A", transform=ax.transAxes, fontsize=pts["head"], fontweight="bold", va="bottom")

    rs = rl["resplit"]["kmer_lgbm"]
    v, pub = np.asarray(rs["values"]), rs["published"]
    lo, hi = min(v.min(), pub), max(v.max(), pub)
    pad = 0.15 * max(hi - lo, 0.02)
    # bins span exactly the axis, so that no bar is drawn (and clipped) outside it
    counts, _, _ = bx.hist(v, bins=24, range=(lo - pad, hi + pad), color="#BBBBBB", edgecolor="white",
                           linewidth=0.4)
    bx.set_xlim(lo - pad, hi + pad)
    # the label sits in headroom above the tallest bar, on top of a line that stops short of it, so that
    # it can touch neither a bar nor the line wherever the published split falls in the distribution
    peak = counts.max()
    bx.set_ylim(0, peak * 1.45)
    bx.vlines(pub, 0, peak * 1.12, color=OKABE_ITO["vermillion"], lw=1.0)
    where = (pub - (lo - pad)) / (hi - lo + 2 * pad)
    bx.text(pub, peak * 1.16, "published split", color=OKABE_ITO["vermillion"], fontsize=pts["small"],
            va="bottom", ha="right" if where > 0.75 else "left" if where < 0.25 else "center")
    bx.set_xlabel("RNAlight AUROC,\n200 random splits (Q4)", fontsize=pts["label"])
    bx.set_ylabel("splits", fontsize=pts["label"])
    bx.text(-0.22, 1.06, "B", transform=bx.transAxes, fontsize=pts["head"], fontweight="bold", va="bottom")

    fig.tight_layout(rect=rect)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    fig = figure(args.inp)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print("wrote", *savefig(fig, args.out), save_tif(fig, args.out))


if __name__ == "__main__":
    main()
