#!/usr/bin/env python
"""Main figures for the lncRNA leakage/confound audit.

Provenance: Fig. 2 and Fig. 3 read their DVMnet / LncPTPred values from the results JSON written by
repro/exp_dvmnet_model.py and repro/exp_lncptpred_model.py, so the figures cannot drift from the runs.
RNAlight values are the measured results recorded in notes/RESULT_A1_homology.md (produced by
repro/exp_homology_split.py, exp_composition_confounds.py, exp_confound_matched.py).

Palette: validated for colour-vision deficiency with the dataviz skill's validate_palette.js.
  2-slot set (#2166AC, #B2182B) -> all checks pass (worst adjacent protan dE 21.1).
  3-slot set (#2166AC, #E08214, #B2182B) -> passes with a contrast WARN on the orange, which
  obligates visible direct labels; those are present on every marked value.
A 4-slot set was rejected: tritan dE 3.7 between orange and green.
Secondary encoding is carried by hatching (Fig. 2) and marker shape (Fig. 3), so both figures
survive greyscale printing.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("AUDIT_FIGS", os.path.join(HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)


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
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

BLUE, RED, ORANGE, GREY = "#2166AC", "#B2182B", "#E08214", "#666666"

dvm = {r["condition"]: r for r in json.load(open(results_path("dvmnet_model_results.json")))}
lnc = {r["condition"]: r for r in json.load(open(results_path("lncptpred_model_results.json")))}

# ============ Figure 1 — RNAlight (values from notes/RESULT_A1_homology.md) ============
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.2))
splits = ["reported\nfixed test", "random\nCV", "homology\nid0.5", "homology\nid0.8"]
sv = [0.783, 0.728, 0.728, 0.725]
axA.bar(range(len(sv)), sv, color=[GREY, BLUE, BLUE, BLUE], edgecolor="black", linewidth=0.4)
axA.axhline(0.728, ls="--", lw=0.8, color=BLUE)
axA.set_xticks(range(len(splits)))
axA.set_xticklabels(splits, fontsize=7)
axA.set_ylim(0.45, 0.90)
axA.set_ylabel("AUROC")
axA.set_title("A  Homology-aware split = no-op", fontsize=9, loc="left")
# Headroom to 0.90 so this sits above the bars: at the old ylim it was printed across the
# random-CV and id0.5 bars (verified by repro/check_figure_overlaps.py).
axA.annotate("only 1.6–5.2% of lncRNAs\nhave any homolog", xy=(2.4, 0.735), xytext=(1.25, 0.815),
             fontsize=6.5, ha="left", va="bottom", arrowprops=dict(arrowstyle="->", lw=0.6))

featsets = ["length", "GC", "len+GC", "mono\n(4)", "di\n(16)", "3-mer\n(64)", "full\n(1344)"]
vals = [0.581, 0.562, 0.615, 0.578, 0.703, 0.719, 0.728]
axB.bar(range(len(vals)), vals, color=[GREY] * 4 + [ORANGE, ORANGE, BLUE],
        edgecolor="black", linewidth=0.4)
axB.axhline(0.728, ls="--", lw=0.8, color=BLUE)
axB.axhline(0.5, ls=":", lw=0.8, color="black")
axB.set_xticks(range(len(featsets)))
axB.set_xticklabels(featsets, fontsize=7)
# Same limits as panel A: both panels plot AUROC, so a given value must sit at the same
# height in both. Different limits across panels of one figure invite false comparison.
axB.set_ylim(0.45, 0.90)
axB.set_ylabel("AUROC (5-fold CV)")
axB.set_title("B  RNAlight signal = bulk composition", fontsize=9, loc="left")
axB.annotate("16 dinucleotides\n= 97% of full model", xy=(4, 0.712), xytext=(1.15, 0.815),
             fontsize=6.5, ha="left", va="bottom", arrowprops=dict(arrowstyle="->", lw=0.6))
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_rnalight.png", dpi=300)
fig.savefig(f"{OUT}/fig1_rnalight.pdf")
print(f"wrote {OUT}/fig1_rnalight.png/.pdf")

# ============ Figure 2 — DVMnet deep model under protocol ablation ============
ORDER = ["verbatim", "random-uniform", "random-degmat", "cold-uniform", "cold-degmat"]
LABEL = ["author\nprotocol", "valid pair\ntypes only", "+ degree-\nmatched neg",
         "cold-start\n+ uniform", "cold-start\n+ deg-matched"]
best = [dvm[c]["best_mean"] for c in ORDER]
bsd = [dvm[c]["best_std"] for c in ORDER]
final = [dvm[c]["final_mean"] for c in ORDER]
fsd = [dvm[c]["final_std"] for c in ORDER]
degb = [dvm[c]["degree_baseline_mean"] for c in ORDER]

fig2, (ax2A, ax2B) = plt.subplots(1, 2, figsize=(7.2, 3.6),
                                  gridspec_kw={"width_ratios": [1.8, 1]})
x = np.arange(len(ORDER))
# Dot plot, not bars: AUROC's meaningful reference is chance (0.5), not zero, so a bar's
# area-from-zero would either mislead or force a truncated baseline. Dots carry position only.
for xi, b, f in zip(x, best, final):
    ax2A.plot([xi, xi], [f, b], color="#BBBBBB", lw=1.4, zorder=1, solid_capstyle="round")
ax2A.errorbar(x, best, yerr=bsd, fmt="o", ms=7, color=BLUE, mec="black", mew=0.4,
              capsize=2.5, elinewidth=0.7, zorder=3, label="deep model, best epoch")
ax2A.errorbar(x, final, yerr=fsd, fmt="s", ms=6.5, mfc="white", mec=RED, mew=1.4,
              ecolor=RED, capsize=2.5, elinewidth=0.7, zorder=3, label="deep model, final epoch")
ax2A.plot(x, degb, marker="_", ms=14, mew=1.8, color="black", ls="none", zorder=4,
          label="degree-sum baseline (0 parameters)")
ax2A.axhline(0.87, ls="--", lw=0.9, color=GREY)
ax2A.text(4.42, 0.879, "reported 0.87", fontsize=6.5, ha="right", color=GREY)
ax2A.axhline(0.5, ls=":", lw=0.9, color="black")
ax2A.text(4.42, 0.508, "chance", fontsize=6.5, ha="right")
ax2A.set_xticks(x)
ax2A.set_xticklabels(LABEL, fontsize=6.4)
ax2A.set_xlim(-0.55, 4.55)
ax2A.set_ylim(0.33, 0.95)
ax2A.set_ylabel("AUROC")
ax2A.set_title("A  The reported 0.87 is unattainable under every control we tested",
               fontsize=8.5, loc="left")
ax2A.legend(fontsize=6.2, loc="lower left", framealpha=0.95)
ax2A.annotate("fixing only the negatives —\nsplit left exactly as published —\ncollapses it to chance",
              xy=(2.10, 0.522), xytext=(2.92, 0.615), fontsize=6.3, ha="right", va="bottom",
              arrowprops=dict(arrowstyle="->", lw=0.6, shrinkA=2, shrinkB=3))
ax2A.grid(axis="y", lw=0.3, color="#EEEEEE", zorder=0)
ax2A.set_axisbelow(True)

gap = [b - f for b, f in zip(best, final)]
# paired-test significance from repro/exp_dvmnet_stats.py (n=5 folds, paired t)
SIG = [False, True, False, False, True]
ax2B.bar(x, gap, 0.6, color=ORANGE, edgecolor="black", linewidth=0.4)
for xi, g, sig in zip(x, gap, SIG):
    lab = f"{g:+.3f}" if sig else f"{g:+.3f}\nn.s."
    ax2B.text(xi, g + 0.005, lab, ha="center", fontsize=6.0, linespacing=1.15)
ax2B.set_xticks(x)
ax2B.set_xticklabels(["verbatim", "valid\ntypes", "+deg-\nmatched", "cold\n+unif", "cold\n+deg"],
                     fontsize=6.3)
ax2B.set_ylim(0, 0.235)
ax2B.set_ylabel("AUROC gained by choosing the\nbest epoch on the evaluation fold", fontsize=7.5)
ax2B.set_title("B  Worst under entity-disjoint\nsplits, not under the leakiest", fontsize=8.5,
               loc="left")
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig2_dvmnet.png", dpi=300)
fig2.savefig(f"{OUT}/fig2_dvmnet.pdf")
print(f"wrote {OUT}/fig2_dvmnet.png/.pdf")

# ============ Figure 3 — cross-target synthesis ============
# Per target: reported claim -> our reproduction -> trivial-signal ceiling -> leakage-controlled.
# Marker SHAPE is the primary encoding so the panel survives greyscale; colour is redundant.
targets = [
    dict(name="RNAlight\n(localization)",
         reported=0.783, reported_txt="0.783", reported_open=False,
         repro=0.728, trivial=0.703, trivial_txt="dinucleotide",
         controlled=0.689, controlled_txt="length/GC-matched"),
    dict(name="DVMnet\n(lncRNA–miRNA)",
         reported=0.87, reported_txt="0.87", reported_open=False,
         repro=dvm["verbatim"]["best_mean"], trivial=dvm["verbatim"]["degree_baseline_mean"],
         trivial_txt="degree-sum", controlled=dvm["cold-degmat"]["best_mean"],
         controlled_txt="cold-start +\ndeg-matched"),
    dict(name="LncPTPred\n(lncRNA–protein)",
         reported=0.90, reported_txt="“>0.9”\n(no point estimate\npublished)", reported_open=True,
         repro=lnc["kmer+protein [unmatched (as shipped)]"]["auroc_mean"],
         trivial=lnc["gc-only [unmatched (as shipped)]"]["auroc_mean"], trivial_txt="GC only",
         controlled=lnc["kmer+protein [GC-matched]"]["auroc_mean"], controlled_txt="GC-matched"),
]

fig3, ax3 = plt.subplots(figsize=(7.2, 4.1))
for i, t in enumerate(targets):
    y = len(targets) - 1 - i
    lo, hi = min(t["controlled"], t["reported"]), max(t["controlled"], t["reported"])
    ax3.plot([lo, hi], [y, y], color="#D5D5D5", lw=6, solid_capstyle="round", zorder=1)
    ax3.plot(t["reported"], y, marker="^", ms=9, zorder=3,
             mfc="white" if t["reported_open"] else GREY, mec=GREY, mew=1.2, ls="none")
    ax3.plot(t["repro"], y, marker="o", ms=8, color=BLUE, zorder=3, ls="none")
    ax3.plot(t["trivial"], y, marker="D", ms=7, color=ORANGE, mec="black", mew=0.4, zorder=3,
             ls="none")
    ax3.plot(t["controlled"], y, marker="s", ms=8, color=RED, zorder=3, ls="none")
    ax3.text(t["reported"] + 0.006, y + 0.16, t["reported_txt"], fontsize=6.2, color=GREY,
             va="bottom")
    # Left-aligned just right of its marker, not centred on it: the repro and controlled
    # labels share the row below the line, and centring both collided them wherever the two
    # values sit close together (RNAlight 0.728 vs 0.689; LncPTPred 0.790 vs 0.759).
    ax3.text(t["repro"] + 0.005, y - 0.22, f"{t['repro']:.3f}", fontsize=6.2, color=BLUE,
             ha="left", va="top")
    ax3.text(t["trivial"], y + 0.16, f"{t['trivial']:.3f}\n{t['trivial_txt']}", fontsize=6.0,
             color=ORANGE, ha="center", va="bottom")
    ax3.text(t["controlled"], y - 0.22, f"{t['controlled']:.3f}\n{t['controlled_txt']}",
             fontsize=6.0, color=RED, ha="center", va="top")

ax3.axvline(0.5, ls=":", lw=0.9, color="black")
ax3.text(0.505, -0.72, "chance", fontsize=6.5)
ax3.set_yticks(range(len(targets)))
ax3.set_yticklabels([t["name"] for t in reversed(targets)], fontsize=7.5)
ax3.set_xlim(0.45, 1.03)
ax3.set_ylim(-0.85, 2.6)
ax3.set_xlabel("AUROC")
# pad clears the two-row legend below it; verified by repro/check_figure_overlaps.py, which
# measures the rendered boxes. At pad=34 the legend's top row ran into the title.
ax3.set_title("Reported claim vs trivial-signal ceiling vs leakage-controlled estimate",
              fontsize=9, loc="left", pad=48)
# legend lives outside the data area: every row here carries direct value labels, and an
# in-axes legend covered the LncPTPred row.
ax3.legend(handles=[
    Line2D([0], [0], marker="^", color=GREY, mfc=GREY, ls="none", ms=8, label="reported by authors"),
    Line2D([0], [0], marker="^", color=GREY, mfc="white", ls="none", ms=8,
           label="reported as an inequality only"),
    Line2D([0], [0], marker="o", color=BLUE, ls="none", ms=8, label="our reproduction"),
    Line2D([0], [0], marker="D", color=ORANGE, mec="black", mew=0.4, ls="none", ms=7,
           label="trivial-signal ceiling (0–1 feature,\nunder the model's own protocol)"),
    Line2D([0], [0], marker="s", color=RED, ls="none", ms=8,
           label="leakage-controlled estimate\n(a different protocol — see caption)"),
], fontsize=6.2, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, frameon=False,
   handletextpad=0.4, columnspacing=1.2)
ax3.grid(axis="x", lw=0.3, color="#EEEEEE", zorder=0)
ax3.set_axisbelow(True)
fig3.tight_layout()
fig3.savefig(f"{OUT}/fig3_synthesis.png", dpi=300)
fig3.savefig(f"{OUT}/fig3_synthesis.pdf")
print(f"wrote {OUT}/fig3_synthesis.png/.pdf")
