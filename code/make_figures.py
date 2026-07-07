#!/usr/bin/env python
"""Main figures for the lncRNA leakage/confound audit.
Values are the measured results recorded in notes/ (hard-coded here for the figure;
each traces to a repro/exp_*.py run)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- path config: edit these or set env vars. Defaults assume repo layout code/ + data/ ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
DATA = _os.environ.get('AUDIT_DATA', _os.path.join(_ROOT, 'data'))
FIG  = _os.environ.get('AUDIT_FIGS', _os.path.join(_ROOT, 'figures'))
RNALIGHT_DIR = _os.environ.get('RNALIGHT_DIR', _os.path.join(_ROOT, 'external', 'RNAlight'))
DVMNET_DIR   = _os.environ.get('DVMNET_DIR',   _os.path.join(_ROOT, 'external', 'DVMnet'))
# --- end path config ---
OUT = FIG
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spanned": False} if False else {"font.size": 9})
CB = {"blue":"#2166AC","red":"#B2182B","grey":"#999999","green":"#1B7837","orange":"#E08214"}

# ================= Figure 1 — RNAlight (A = homology no-op, B = composition) =================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.2))

# 1A: AUROC by split (homology no-op)
splits = ["reported\nfixed test","random\nCV","homology\nid0.5","homology\nid0.8"]
sv =     [0.783, 0.728, 0.728, 0.725]
sc = [CB["grey"], CB["blue"], CB["green"], CB["green"]]
axA.bar(range(len(sv)), sv, color=sc, edgecolor="black", linewidth=0.4)
axA.axhline(0.728, ls="--", lw=0.8, color=CB["blue"])
axA.set_xticks(range(len(splits))); axA.set_xticklabels(splits, fontsize=7)
axA.set_ylim(0.45, 0.82); axA.set_ylabel("AUROC")
axA.set_title("A  Homology-aware split = no-op", fontsize=9, loc="left")
axA.annotate("only 1.6–5.2% of lncRNAs\nhave any homolog", xy=(2.5,0.728), xytext=(0.7,0.60),
             fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))

# 1B: AUROC by feature set (bulk composition)
featsets = ["length","GC","len+GC","mono\n(4)","di\n(16)","3-mer\n(64)","full\n(1344)"]
vals =     [0.581, 0.562, 0.615, 0.578, 0.703, 0.719, 0.728]
colors = [CB["grey"]]*4 + [CB["red"], CB["orange"], CB["blue"]]
axB.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.4)
axB.axhline(0.728, ls="--", lw=0.8, color=CB["blue"])
axB.axhline(0.5, ls=":", lw=0.8, color="black")
axB.set_xticks(range(len(featsets))); axB.set_xticklabels(featsets, fontsize=7)
axB.set_ylim(0.45, 0.80); axB.set_ylabel("AUROC (5-fold CV)")
axB.set_title("B  RNAlight signal = bulk composition", fontsize=9, loc="left")
axB.annotate("16 dinucleotides\n= 97% of full model", xy=(4,0.703), xytext=(1.4,0.755),
             fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_rnalight.png", dpi=300); fig.savefig(f"{OUT}/fig1_rnalight.pdf")
print(f"wrote {OUT}/fig1_rnalight.png/.pdf")

# ================= Figure 2 — DVMnet =================
fig2, (ax2A, ax2B) = plt.subplots(1, 2, figsize=(7.2, 3.2))

# 2A: degree_sum under three regimes
regimes = ["random-edge\n+ uniform neg\n(reconstruction)","random-edge\n+ degree-matched\nneg","cold-start\n(entity-disjoint)"]
rv = [0.750, 0.458, 0.500]
rc = [CB["red"], CB["blue"], CB["blue"]]
ax2A.bar(range(3), rv, color=rc, edgecolor="black", linewidth=0.4)
ax2A.axhline(0.5, ls=":", lw=0.9, color="black"); ax2A.text(2.5, 0.51, "chance", fontsize=6.5, ha="right")
ax2A.set_xticks(range(3)); ax2A.set_xticklabels(regimes, fontsize=6.5)
ax2A.set_ylim(0.40, 0.80); ax2A.set_ylabel("AUROC — degree-sum heuristic\n(zero parameters)")
ax2A.set_title("A  Degree-sum heuristic is predictive\nunder the random-edge protocol", fontsize=8.5, loc="left")

# 2B: all heuristics random-edge vs cold-start
heur = ["degree_sum","resource_alloc","pref_attach"]
rnd = [0.750, 0.630, 0.569]; cold = [0.500, 0.500, 0.500]
x = np.arange(len(heur)); w = 0.38
ax2B.bar(x-w/2, rnd, w, label="random-edge + uniform neg", color=CB["red"], edgecolor="black", linewidth=0.4)
ax2B.bar(x+w/2, cold, w, label="entity-disjoint cold-start", color=CB["blue"], edgecolor="black", linewidth=0.4)
ax2B.axhline(0.5, ls=":", lw=0.9, color="black")
ax2B.set_xticks(x); ax2B.set_xticklabels(heur, fontsize=7); ax2B.set_ylim(0.40, 0.80)
ax2B.set_ylabel("AUROC"); ax2B.legend(fontsize=6.5, loc="upper right")
ax2B.set_title("B  Topology heuristics fall to chance\nunder entity-disjoint splits", fontsize=8.5, loc="left")
fig2.tight_layout(); fig2.savefig(f"{OUT}/fig2_dvmnet.png", dpi=300); fig2.savefig(f"{OUT}/fig2_dvmnet.pdf")
print(f"wrote {OUT}/fig2_dvmnet.png/.pdf")
