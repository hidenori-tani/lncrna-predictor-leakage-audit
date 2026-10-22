#!/usr/bin/env python
"""Figure 3 (synthesis), rebuilt for the revision.

Only the sources change.  DVMnet's three points now come from the repeated-CV run, and
LncPTPred's from the reconstruction built out of the authors' released feature code; RNAlight's
analyses are unchanged and so are its values.  Layout, palette and encodings are as submitted.
"""
import json
import os
import shutil
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPRO = os.path.join(ROOT, "repro")
FIGS = os.path.join(ROOT, "figures")
TIFF_DIR = os.path.join(HERE, "figures")

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     # Type3 は多くの誌の自動チェックで落ちる。TrueType で埋め込む。
                     "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, RED, ORANGE, GREY = "#2166AC", "#B2182B", "#E08214", "#666666"

# ScholarOne（SAGMB）は図を拡大縮小せず、著者の版面の上に絶対位置で3辺に刷る。
# 実測（2026-09-03・SAGMB.2026.0060.R1 の proof 図3頁）＝
#   上 y 3.3–7.5 mm（走り頭・頁番号）／下 端から 10.4–14.7 mm（走り頭）／
#   左 x 2.8–6.4 mm（行番号・頁の全高）。安全余裕 1.5 mm を足して空ける。
# 図そのものは拡大縮小せず、版面を広げて内容を寄せるだけ（文字サイズ・線幅は不変）。
S1_PAD_MM = {"left": 5.0, "top": 6.5, "bottom": 14.5}
MM_PER_IN = 25.4


def layout_for_scholarone(fig):
    """版面を広げ、元の描画領域をそのまま余白の内側に置く。"""
    w_in, h_in = fig.get_size_inches()
    l = S1_PAD_MM["left"] / MM_PER_IN
    t = S1_PAD_MM["top"] / MM_PER_IN
    b = S1_PAD_MM["bottom"] / MM_PER_IN
    fig.set_size_inches(w_in + l, h_in + t + b)
    W, H = fig.get_size_inches()
    fig.tight_layout(rect=[l / W, b / H, 1.0, 1.0 - t / H])



def load(name):
    p = os.path.join(REPRO, name)
    if not os.path.exists(p):
        sys.exit(f"結果がまだ無い: {p}")
    return json.load(open(p))


rev = {r["condition"]: r["summary"] for r in load("dvmnet_revision_results.json")}
sens = {r["condition"]: r["summary"] for r in load("dvmnet_sensitivity_results.json")}
lex = {r["condition"]: r for r in load("lncptpred_exact_results.json")}
lnc_old = {r["condition"]: r for r in load("lncptpred_model_results.json")}

need = ["exact lncRNA 351 + protein 369", "exact full [GC-matched]"]
for k in need:
    if k not in lex:
        sys.exit(f"LncPTPred の条件が足りない: {k}")

targets = [
    dict(name="RNAlight\n(localization)",
         reported=0.783, reported_txt="0.783", reported_open=False,
         repro=0.728, trivial=0.703, trivial_txt="dinucleotide",
         controlled=0.689, controlled_txt="length/GC-matched"),
    dict(name="DVMnet\n(lncRNA–miRNA)",
         reported=0.87, reported_txt="0.87", reported_open=False,
         repro=rev["random-verbatim"]["full_best"]["mean"],
         trivial=rev["random-verbatim"]["degree_test"]["mean"],
         trivial_txt="degree-sum",
         controlled=rev["cold-degmat"]["nested"]["mean"],
         controlled_txt="cold-start +\ndeg-matched", controlled_ha="left",
         # Reviewer 2, point 1: the square is the strictest member of a family of defensible
         # negative-sampling schemes, so the row also carries the family's range (Table 8).
         controlled_range=[m["nested"]["mean"] for c, m in
                           list(rev.items()) + list(sens.items())
                           if c.startswith("cold-")]),
    dict(name="LncPTPred\n(lncRNA–protein)",
         reported=0.90, reported_txt="“>0.9”\n(no point estimate\npublished)", reported_open=True,
         repro=lex["exact lncRNA 351 + protein 369"]["auroc_mean"],
         trivial=lnc_old["gc-only [unmatched (as shipped)]"]["auroc_mean"], trivial_txt="GC only",
         controlled=lex["exact full [GC-matched]"]["auroc_mean"], controlled_txt="GC-matched"),
]

fig3, ax3 = plt.subplots(figsize=(7.2, 5.3))
for i, t in enumerate(targets):
    y = len(targets) - 1 - i
    lo, hi = min(t["controlled"], t["reported"]), max(t["controlled"], t["reported"])
    ax3.plot([lo, hi], [y, y], color="#D5D5D5", lw=6, solid_capstyle="round", zorder=1)
    if t.get("controlled_range"):
        a, b = min(t["controlled_range"]), max(t["controlled_range"])
        ax3.plot([a, b], [y, y], color=RED, lw=1.6, alpha=0.55, solid_capstyle="butt", zorder=2)
        for x in (a, b):
            ax3.plot([x, x], [y - 0.07, y + 0.07], color=RED, lw=1.6, alpha=0.55, zorder=2)
    ax3.plot(t["reported"], y, marker="^", ms=9, zorder=3,
             mfc="white" if t["reported_open"] else GREY, mec=GREY, mew=1.2, ls="none")
    ax3.plot(t["repro"], y, marker="o", ms=8, color=BLUE, zorder=3, ls="none")
    ax3.plot(t["trivial"], y, marker="D", ms=7, color=ORANGE, mec="black", mew=0.4, zorder=3,
             ls="none")
    ax3.plot(t["controlled"], y, marker="s", ms=8, color=RED, zorder=3, ls="none")
    ax3.text(t["reported"] + 0.006, y + 0.16, t["reported_txt"], fontsize=7.7, color=GREY,
             va="bottom")
    ax3.text(t["repro"], y + 0.44, f"{t['repro']:.3f}", fontsize=7.7, color=BLUE,
             ha="center", va="bottom")
    ax3.text(t["trivial"], y + 0.16, f"{t['trivial']:.3f}\n{t['trivial_txt']}", fontsize=7.7,
             color=ORANGE, ha="center", va="bottom")
    cha = t.get("controlled_ha", "center")
    ax3.text(t["controlled"] + (0.009 if cha == "left" else 0.0), y - 0.22,
             f"{t['controlled']:.3f}\n{t['controlled_txt']}",
             fontsize=7.7, color=RED, ha=cha, va="top")

ax3.axvline(0.5, ls=":", lw=0.9, color="black")
ax3.text(0.505, -0.72, "chance", fontsize=7.7)
ax3.set_yticks(range(len(targets)))
ax3.set_yticklabels([t["name"] for t in reversed(targets)], fontsize=7.7)
ax3.set_xlim(0.45, 1.03)
ax3.set_ylim(-0.95, 2.78)
ax3.set_xlabel("AUROC")
ax3.set_title("Reported claim vs trivial-signal ceiling vs leakage-controlled estimate",
              fontsize=9, loc="left", pad=48)
ax3.legend(handles=[
    Line2D([0], [0], marker="^", color=GREY, mfc=GREY, ls="none", ms=8, label="reported by authors"),
    Line2D([0], [0], marker="^", color=GREY, mfc="white", ls="none", ms=8,
           label="reported as an inequality only"),
    Line2D([0], [0], marker="o", color=BLUE, ls="none", ms=8, label="our reproduction"),
    Line2D([0], [0], marker="D", color=ORANGE, mec="black", mew=0.4, ls="none", ms=7,
           label="trivial-signal ceiling (0–1 feature,\nunder the model's own protocol)"),
    Line2D([0], [0], marker="s", color=RED, ls="none", ms=8,
           label="leakage-controlled estimate\n(protocol named beside it)"),
    Line2D([0], [0], color=RED, lw=1.6, alpha=0.55,
           label="range over negative-sampling\nschemes (DVMnet, Table 8)"),
], fontsize=7.6, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, frameon=False,
   handletextpad=0.4, columnspacing=1.2)
ax3.grid(axis="x", lw=0.3, color="#EEEEEE", zorder=0)
ax3.set_axisbelow(True)
fig3.tight_layout()
layout_for_scholarone(fig3)

def save_tif(fig, path):
    """TIF を白地の RGB で保存する。

    matplotlib が書く TIFF は RGBA（透明つき）で、組版の RIP によっては透明部分が黒く
    刷られる。誌の規定は「JPG/TIF/EPS・300 dpi 以上」だけで透明には触れていないので、
    こちらで潰しておく（2026-09-03 実測：3点とも RGBA だった）。
    """
    from PIL import Image
    fig.savefig(path, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        bg.save(path, dpi=(300, 300), compression="tiff_lzw")

os.makedirs(TIFF_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for ext in ("png", "pdf"):
    dst = os.path.join(FIGS, f"fig3_synthesis.{ext}")
    if os.path.exists(dst):
        shutil.copy2(dst, dst + f".bak_{stamp}")
    fig3.savefig(dst, dpi=300)
save_tif(fig3, os.path.join(TIFF_DIR, "Figure_3.tif"))
# 誌の規定は JPG/TIF/EPS だが、先生が中身を確認しやすいようにベクタの PDF も併置する
# （提出は TIF。PDF は規定に無いので投稿画面には出さない）。
fig3.savefig(os.path.join(TIFF_DIR, "Figure_3.pdf"))
print(f"  ✔ figures/fig3_synthesis.png/.pdf（旧版は .bak_{stamp} に退避）")
for t in targets:
    print(f"  {t['name'].splitlines()[0]:<11} reported {t['reported']:.3f}  repro {t['repro']:.3f}"
          f"  trivial {t['trivial']:.3f}  controlled {t['controlled']:.3f}")
