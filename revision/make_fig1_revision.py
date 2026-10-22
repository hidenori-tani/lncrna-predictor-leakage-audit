#!/usr/bin/env python
"""Figure 1 (RNAlight), rebuilt for the revision.

The RNAlight analyses are unchanged by this revision and so are every value here; what changes is
the typography.  The submitted PDF embedded Type3 fonts, its smallest label measured 6.2 pt, and
the overlap gate found ten collisions between tick labels and the axis spine.  This script fixes
those three things and nothing else, so the figure still shows exactly what it showed.

It is separate from repro/make_figures.py deliberately: that script writes all three figures, and
running it now would overwrite the revised Figures 2 and 3 with their submitted versions.

Values: notes/RESULT_A1_homology.md (repro/exp_homology_split.py, exp_composition_confounds.py,
exp_confound_matched.py).
"""
import os
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "figures")
TIFF_DIR = os.path.join(HERE, "figures")

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     # Type3 は多くの誌の自動チェックで落ちる。TrueType で埋め込む。
                     "pdf.fonttype": 42, "ps.fonttype": 42})
YLIM0 = 0.45          # 軸の下端。棒の底をここに置く（下記）
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


TICK_FS, ANNOT_FS, TICK_PAD = 7.7, 7.7, 5.0

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.35))

splits = ["reported\nfixed test", "random\nCV", "homology\nid0.5", "homology\nid0.8"]
sv = [0.783, 0.728, 0.728, 0.725]
axA.bar(range(len(sv)), [v - YLIM0 for v in sv], bottom=YLIM0,
        color=[GREY, BLUE, BLUE, BLUE], edgecolor="black", linewidth=0.4)
axA.axhline(0.728, ls="--", lw=0.8, color=BLUE)
axA.set_xticks(range(len(splits)))
axA.set_xticklabels(splits, fontsize=TICK_FS)
axA.set_ylim(YLIM0, 0.90)
axA.set_ylabel("AUROC", labelpad=6)
axA.set_title("A  Homology-aware split = no-op", fontsize=9, loc="left", pad=8)
# Headroom to 0.90 so this sits above the bars: at the old ylim it was printed across the
# random-CV and id0.5 bars (verified by repro/check_figure_overlaps.py).
axA.annotate("only 1.6–5.2% of lncRNAs\nhave any homolog", xy=(2.4, 0.735), xytext=(1.10, 0.822),
             fontsize=ANNOT_FS, ha="left", va="bottom", arrowprops=dict(arrowstyle="->", lw=0.6))

# 7.5 pt にすると "len+GC" が隣の "mono" に触れる（重なりゲートで実測）。2行に割って幅を詰める。
featsets = ["length", "GC", "len\n+GC", "mono\n(4)", "di\n(16)", "3-mer\n(64)", "full\n(1344)"]
vals = [0.581, 0.562, 0.615, 0.578, 0.703, 0.719, 0.728]
axB.bar(range(len(vals)), [v - YLIM0 for v in vals], bottom=YLIM0,
        color=[GREY] * 4 + [ORANGE, ORANGE, BLUE], edgecolor="black", linewidth=0.4)
axB.axhline(0.728, ls="--", lw=0.8, color=BLUE)
axB.axhline(0.5, ls=":", lw=0.8, color="black")
axB.set_xticks(range(len(featsets)))
axB.set_xticklabels(featsets, fontsize=TICK_FS)
# Same limits as panel A: both panels plot AUROC, so a given value must sit at the same
# height in both. Different limits across panels of one figure invite false comparison.
axB.set_ylim(YLIM0, 0.90)
axB.set_ylabel("AUROC (5-fold CV)", labelpad=6)
axB.set_title("B  RNAlight signal = bulk composition", fontsize=9, loc="left", pad=8)
axB.annotate("16 dinucleotides\n= 97% of full model", xy=(4, 0.712), xytext=(0.95, 0.822),
             fontsize=ANNOT_FS, ha="left", va="bottom", arrowprops=dict(arrowstyle="->", lw=0.6))

for ax in (axA, axB):
    # 投稿版では x 目盛ラベルが軸線に接していた（重なりゲートで10件 FAIL）。離す。
    ax.tick_params(axis="x", pad=TICK_PAD, labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)

fig.tight_layout()
layout_for_scholarone(fig)

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
    dst = os.path.join(FIGS, f"fig1_rnalight.{ext}")
    if os.path.exists(dst):
        shutil.copy2(dst, dst + f".bak_{stamp}")
    fig.savefig(dst, dpi=300)
save_tif(fig, os.path.join(TIFF_DIR, "Figure_1.tif"))
# 誌の規定は JPG/TIF/EPS だが、先生が中身を確認しやすいようにベクタの PDF も併置する
# （提出は TIF。PDF は規定に無いので投稿画面には出さない）。
fig.savefig(os.path.join(TIFF_DIR, "Figure_1.pdf"))
print(f"  ✔ figures/fig1_rnalight.png/.pdf（旧版は .bak_{stamp} に退避）")
