#!/usr/bin/env python
"""Figure 1 (R2): every model family on the DVMnet benchmark, under each condition, by question.

Reviewer 2, point 1: "compare several baseline models on the same task and dataset". The figure
shows, for one benchmark, what each protocol change does to models whose information is known
(grey: node-level information only; blue: models that read the pair) beside what it does to DVMnet
(vermillion). Error bars are the SD over independent cross-validation partitions.

Provenance: every value comes from revision_r2/results/dvmnet_baselines.json, which carries the
baselines and, fold by fold, DVMnet's nested estimate from the R1 run (released, uniform) or the R2
re-run with training negatives drawn from the training graph (every other condition).
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

HEIGHT_MM = 92.0

CONDITIONS = [("random", "verbatim", "released", "Q1"),
              ("random", "uniform", "valid pairs\nonly", "Q4"),
              ("random", "swap", "swap", "Q2"),
              ("cold", "uniform", "unseen lncRNAs,\nuniform", "Q3"),
              ("cold", "swap", "unseen lncRNAs,\nswap", "Q3 and Q2")]
# (key, label, colour, marker, filled)
FAMILIES = [("degree_sum", "degree sum", "#4D4D4D", "o", False),
            ("popularity_lr", "popularity", "#4D4D4D", "s", False),
            ("node_id_lr", "node identity", "#4D4D4D", "^", True),
            ("composition_gbm", "composition", "#8C8C8C", "D", True),
            ("doc2vec_gbm", "doc2vec", "#8C8C8C", "v", True),
            ("seed_sites", "seed sites", OKABE_ITO["blue"], "o", True),
            ("seed_family_lr", "seed-site family", OKABE_ITO["blue"], "s", False),
            ("DVMnet", "DVMnet", OKABE_ITO["vermillion"], "o", True)]


def value(summary, key):
    """Prefer the training-graph-only scope where it exists (identical by construction under the
    random split with uniform negatives; the only valid one for every other matched condition)."""
    for k in (key + "@trainonly", key):
        if k in summary:
            return summary[k]["mean"], summary[k]["sd"]
    return None


IN = os.path.join(HERE, "results", "dvmnet_baselines.json")
OUT = os.path.join(HERE, "figures", "Figure_1")


def figure(inp=IN):
    """Draw and return the figure; saving is main()'s (check_axes_clipping.py measures the return)."""
    res = {(r["split"], r["negatives"]): r for r in json.load(open(inp))["results"]}
    missing = [f"{s}-{n}" for s, n, _, _ in CONDITIONS if (s, n) not in res]
    if missing:
        sys.exit(f"conditions missing from {inp}: {missing}")

    use_journal_style(JOURNAL)
    pts = pt_scale(JOURNAL)
    fig, ax = plt.subplots(figsize=(mm(WIDTH_MM), mm(HEIGHT_MM)))
    rect = layout_for_scholarone(fig)

    n_f = len(FAMILIES)
    offsets = np.linspace(-0.36, 0.36, n_f)
    for ci, (split, neg, _label, _q) in enumerate(CONDITIONS):
        s = res[(split, neg)]["summary"]
        for fi, (key, _lab, colour, marker, filled) in enumerate(FAMILIES):
            v = value(s, key)
            if v is None:
                sys.exit(f"{key} missing under {split}-{neg}")
            m, sd = v
            size = 5.0 if key == "DVMnet" else 3.6
            ax.errorbar(ci + offsets[fi], m, yerr=sd, fmt=marker, ms=size, color=colour,
                        mfc=colour if filled else "white", mec=colour, mew=0.8, elinewidth=0.6,
                        capsize=0, zorder=3 if key == "DVMnet" else 2)
    ax.axhline(0.5, color="#999999", lw=0.6, ls=":", zorder=1)
    for ci in range(1, len(CONDITIONS)):
        ax.axvline(ci - 0.5, color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)
    ax.set_ylim(0.40, 1.0)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("AUROC (mean ± SD over partitions)", fontsize=pts["label"])
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([c[2] for c in CONDITIONS], fontsize=pts["small"])
    for ci, (_s, _n, _l, q) in enumerate(CONDITIONS):
        ax.text(ci, 0.985, q, ha="center", va="top", fontsize=pts["body"], color="#333333")
    handles = [plt.Line2D([], [], ls="none", marker=mk, ms=5.0 if k == "DVMnet" else 3.6, color=c,
                          mfc=c if fl else "white", mec=c, mew=0.8, label=lab)
               for k, lab, c, mk, fl in FAMILIES]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=pts["small"], handletextpad=0.3, columnspacing=1.2, borderaxespad=0.0)
    fig.tight_layout(rect=rect)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    fig = figure(args.inp)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = savefig(fig, args.out) + [save_tif(fig, args.out)]
    print("wrote", *written)


if __name__ == "__main__":
    main()
