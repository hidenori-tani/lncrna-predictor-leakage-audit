#!/usr/bin/env python
"""Figure 2, rebuilt for the revision.

Panel A now shows the nested estimate (epoch chosen on a held-out selection split, scored on the
disjoint test split) and the oracle (epoch chosen on the test split itself), instead of best-epoch
and final-epoch.  Panel B shows the selection bias measured against that independent reference
instead of (best - final), which Reviewer 2 correctly identified as not a valid estimate of it.
Error bars are the SD over independent CV partitions, which is what repeated cross-validation gives.

Palette and secondary encodings are unchanged from the submitted figure, and the annotation that
used the word "collapses" is reworded with the rest of the manuscript.

Provenance: every value comes from repro/dvmnet_revision_results.json.
"""
import json
import os
import shutil
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "repro", "dvmnet_revision_results.json")
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


ORDER = ["random-verbatim", "random-uniform", "random-degmat", "cold-uniform", "cold-degmat"]
# ラベルは短く保つ：長い版は 3 番目と 4 番目の2行目どうしが重なった（実測 2026-09-03）
# 7.4 pt では隣どうしが横に触れる（重なりゲートで実測）。最後の2つを3行に割る。
LABEL = ["verbatim\n(released)", "valid pair\ntypes", "+ degree-\nmatched",
         "cold-start\n+ uniform", "cold-start\n+ deg-\nmatched"]
SHORT = ["verbatim", "valid\ntypes", "+deg-\nmatched", "cold\n+unif", "cold\n+deg"]

if not os.path.exists(RESULTS):
    sys.exit(f"結果がまだ無い: {RESULTS}")
rev = {r["condition"]: r for r in json.load(open(RESULTS))}
missing = [c for c in ORDER if c not in rev]
if missing:
    sys.exit(f"Stage A が未完了（欠けている条件: {missing}）")

S = {c: rev[c]["summary"] for c in ORDER}
nested = [S[c]["nested"]["mean"] for c in ORDER]
nsd = [S[c]["nested"]["sd"] for c in ORDER]
oracle = [S[c]["test_oracle"]["mean"] for c in ORDER]
osd = [S[c]["test_oracle"]["sd"] for c in ORDER]
degb = [S[c]["degree_test"]["mean"] for c in ORDER]
bias = [S[c]["selection_bias"]["mean"] for c in ORDER]

# significance of the selection bias, paired over partitions (not folds)
SIG = []
for c in ORDER:
    a = S[c]["test_oracle"]["values"]
    b = S[c]["nested"]["values"]
    if len(a) > 1:
        _, p = stats.ttest_rel(a, b)
        SIG.append(bool(p < 0.05))
    else:
        SIG.append(False)

fig2, (ax2A, ax2B) = plt.subplots(1, 2, figsize=(7.2, 4.5),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
x = np.arange(len(ORDER))
for xi, lo, hi in zip(x, nested, oracle):
    ax2A.plot([xi, xi], [lo, hi], color="#BBBBBB", lw=1.4, zorder=1, solid_capstyle="round")
ax2A.errorbar(x, nested, yerr=nsd, fmt="o", ms=7, color=BLUE, mec="black", mew=0.4,
              capsize=2.5, elinewidth=0.7, zorder=3,
              label="epoch chosen on a held-out split")
ax2A.errorbar(x, oracle, yerr=osd, fmt="s", ms=6.5, mfc="white", mec=RED, mew=1.4,
              ecolor=RED, capsize=2.5, elinewidth=0.7, zorder=3,
              label="epoch chosen on the test split (upper bound)")
ax2A.plot(x, degb, marker="_", ms=14, mew=1.8, color="black", ls="none", zorder=4,
          label="degree-sum baseline (0 param.)")
ax2A.axhline(0.87, ls="--", lw=0.9, color=GREY)
ax2A.text(4.42, 0.876, "reported 0.87", fontsize=7.7, ha="right", va="bottom", color=GREY)
ax2A.axhline(0.5, ls=":", lw=0.9, color="black")
ax2A.text(-0.45, 0.512, "chance", fontsize=7.7, ha="left", va="bottom")
ax2A.set_xticks(x)
ax2A.set_xticklabels(LABEL, fontsize=7.6)
ax2A.set_xlim(-0.55, 4.55)
lo_y = min(min(nested) - max(nsd), min(degb)) - 0.04
ax2A.set_ylim(min(0.33, lo_y), 0.95)
ax2A.set_ylabel("AUROC")
ax2A.set_title("A  The reported 0.87 is not attainable under any control we tested",
               fontsize=8.5, loc="left", pad=8)

ax2A.annotate("correcting only the negatives —\nsplit left exactly as published —\n"
              "reduces it to chance",
              # 右上に置くと cold-start 側のマーカーに接する（実測 2026-09-03）。
              # 左上の空白域へ寄せ、矢印で対象を指す。
              xy=(2.02, nested[2] + 0.022), xytext=(2.72, nested[2] + 0.135),
              fontsize=7.6, ha="right", va="bottom", linespacing=1.8,
              arrowprops=dict(arrowstyle="->", lw=0.6, shrinkA=2, shrinkB=3))
ax2A.set_axisbelow(True)

ax2B.bar(x, bias, 0.6, color=ORANGE, edgecolor="black", linewidth=0.4)
top = max(max(bias), 0.02)
for xi, g, sig in zip(x, bias, SIG):
    lab = f"{g:+.3f}" if sig else f"{g:+.3f}\nn.s."
    ax2B.text(xi, g + top * 0.09, lab, ha="center", fontsize=7.6, linespacing=1.45)
ax2B.set_xticks(x)
ax2B.set_xticklabels(SHORT, fontsize=7.6)
ax2B.set_ylim(0, top * 1.35)
ax2B.set_ylabel("AUROC recovered by choosing the epoch\non the test split instead of a held-out one",
                fontsize=7.6, linespacing=1.75)
ax2B.set_title("B  Checkpoint selection,\nmeasured independently", fontsize=8.5,
               loc="left", pad=8, linespacing=1.45)
_h, _l = ax2A.get_legend_handles_labels()
fig2.legend(_h, _l, fontsize=7.6, loc="lower center", ncol=3, frameon=False,
            # 版面の中央でなく「左余白を除いた描画領域」の中央に置く。
            # 0.5 のままだと凡例の先頭のマーカーが ScholarOne の行番号の帯に入る（実測 608 画素）。
            bbox_to_anchor=(0.5 + S1_PAD_MM["left"] / (2 * 187.9), 0.0),
            handletextpad=0.6, columnspacing=1.1)
fig2.tight_layout(rect=[0, 0.075, 1, 1])
layout_for_scholarone(fig2)

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
    dst = os.path.join(FIGS, f"fig2_dvmnet.{ext}")
    if os.path.exists(dst):
        shutil.copy2(dst, dst + f".bak_{stamp}")
    fig2.savefig(dst, dpi=300)
save_tif(fig2, os.path.join(TIFF_DIR, "Figure_2.tif"))
# 誌の規定は JPG/TIF/EPS だが、先生が中身を確認しやすいようにベクタの PDF も併置する
# （提出は TIF。PDF は規定に無いので投稿画面には出さない）。
fig2.savefig(os.path.join(TIFF_DIR, "Figure_2.pdf"))
print(f"  ✔ figures/fig2_dvmnet.png/.pdf（旧版は .bak_{stamp} に退避）")
print(f"  ✔ {TIFF_DIR}/Figure_2.tif")
print("\n=== 図に載った値 ===")
for c, n, o, b, s in zip(ORDER, nested, oracle, bias, SIG):
    print(f"  {c:<17} nested {n:.4f}  oracle {o:.4f}  bias {b:+.4f}{'' if s else '  n.s.'}")
