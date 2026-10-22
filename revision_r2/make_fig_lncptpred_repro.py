#!/usr/bin/env python
"""Figure 3 (R2): LncPTPred reproduced with the full dataset and the original training settings.

(A) Validation log-loss against boosting iteration: the authors' own record, decoded from their
released trained model, and our retraining with the recorded settings on all 1,590,016 rows.
(B) AUROC on the validation split for the settings x data-size combinations that separate the cause
of the earlier gap, with the paper's ">0.9" shown as a region rather than a point.

Provenance: the authors' curve is decoded from models/models.zip by
revision_r2/inspect_released_lncptpred_model.py into external/LncPTPred_released_models/derived/
(not redistributed); ours is repro/lncptpred_full/curves_original.npy; AUROC values come from
revision_r2/results/lncptpred_*.json.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from figcommon_r2 import JOURNAL, WIDTH_MM, layout_for_scholarone, save_tif  # noqa: E402
from figstyle import OKABE_ITO, mm, pt_scale, savefig, use_journal_style  # noqa: E402

HEIGHT_MM = 70.0
DERIVED = os.path.join(ROOT, "external", "LncPTPred_released_models", "derived")
OURS = os.path.join(ROOT, "repro", "lncptpred_full", "curves_original.npy")
RES = os.path.join(HERE, "results")


OUT = os.path.join(HERE, "figures", "Figure_3")


def figure():
    """Draw and return the figure; saving is main()'s (check_axes_clipping.py measures the return)."""
    theirs = np.load(os.path.join(DERIVED, "curve_valid_1_binary_logloss.npy"), allow_pickle=False)
    cab = np.load(OURS, allow_pickle=False)
    rt = json.load(open(os.path.join(RES, "lncptpred_retrain_original.json")))
    nA, nB = rt["n_A"], rt["n_B"]
    ours = (nA * cab[:, 0] + nB * cab[:, 1]) / (nA + nB)
    r1full = json.load(open(os.path.join(RES, "lncptpred_r1_settings_full.json")))
    o400 = json.load(open(os.path.join(RES, "lncptpred_original_n400k.json")))

    use_journal_style(JOURNAL)
    pts = pt_scale(JOURNAL)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(mm(WIDTH_MM), mm(HEIGHT_MM)),
                                 gridspec_kw=dict(width_ratios=[1.5, 1.0]))
    rect = layout_for_scholarone(fig)

    it_t = np.arange(1, len(theirs) + 1)
    it_o = np.arange(1, len(ours) + 1)
    ax.plot(it_t, theirs, color="#333333", lw=1.0, label="authors' record (released model)")
    ax.plot(it_o, ours, color=OKABE_ITO["vermillion"], lw=1.0, ls="--", label="our retraining, full data")
    kt = int(np.argmin(theirs)) + 1
    ko = rt["early_stopping"]["union"]["best"]
    ax.plot([kt], [theirs[kt - 1]], "o", ms=3.4, color="#333333")
    ax.plot([ko], [ours[ko - 1]], "o", ms=3.4, color=OKABE_ITO["vermillion"], mfc="white", mew=0.8)
    ax.set_xscale("log")
    # plain numbers: the mathtext exponents of 10^k are set at 70% of the tick size (5.5 pt < 7 pt)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("boosting iteration", fontsize=pts["label"])
    ax.set_ylabel("validation log-loss", fontsize=pts["label"])
    lo = min(theirs.min(), ours.min())
    ax.set_ylim(lo - 0.02, 0.70)
    # above the axes, where no curve can run through it whatever the data
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, frameon=False, fontsize=pts["small"],
              borderaxespad=0.0, handlelength=1.8)
    ax.text(-0.14, 1.17, "A", transform=ax.transAxes, fontsize=pts["head"], fontweight="bold", va="bottom")

    r1k = json.load(open(os.path.join(ROOT, "revision", "key_numbers.json")))
    r1_400k = r1k["lncptpred_exact"]["exact lncRNA 351 + protein 369"]["auroc_mean"]
    rows = [("earlier settings,\nn = 400,000 (5-fold CV)", r1_400k, "CV"),
            ("earlier settings,\nfull data", r1full["auroc_at_iteration"][str(r1full["iterations_run"])]["union"], ""),
            ("recorded settings,\nn = 400,000", o400["as_released_protocol"], ""),
            ("recorded settings,\nfull data", rt["as_released_protocol"], "")]
    y = np.arange(len(rows))[::-1]
    top = len(rows) - 0.1
    bx.axvspan(0.9, 1.0, facecolor="#EEEEEE", lw=0, zorder=0)
    bx.text(0.95, top - 0.12, 'reported ">0.9"', ha="center", va="top", fontsize=pts["small"],
            color="#555555")
    for yi, (lab, v, _n) in zip(y, rows):
        c = OKABE_ITO["vermillion"] if "recorded settings,\nfull" in lab else "#4D4D4D"
        bx.plot([v], [yi], "o", ms=4.0, color=c)
        # to the right of the point, where the axis has room up to the shaded band (left if it would enter it)
        right = v < 0.86
        bx.annotate(f"{v:.3f}", (v, yi), xytext=(5 if right else -5, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=pts["small"], color=c)
    bx.set_yticks(y)
    bx.set_yticklabels([r[0] for r in rows], fontsize=pts["small"])
    bx.set_xlim(min(0.75, min(r[1] for r in rows) - 0.02), 1.0)
    bx.set_ylim(-0.6, top)
    bx.set_xlabel("AUROC on the validation split", fontsize=pts["label"])
    bx.text(-0.62, 1.17, "B", transform=bx.transAxes, fontsize=pts["head"], fontweight="bold", va="bottom")

    fig.tight_layout(rect=rect)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    fig = figure()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print("wrote", *savefig(fig, args.out), save_tif(fig, args.out))


if __name__ == "__main__":
    main()
