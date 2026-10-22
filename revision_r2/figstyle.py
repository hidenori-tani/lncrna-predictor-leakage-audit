#!/usr/bin/env python3
"""figstyle.py — 投稿可能な図を matplotlib で最初から作るための共有スタイル

なぜ必要か：既存の図生成スクリプトが matplotlib の既定のまま書き出しており、
`check_figure.py` で全 PDF が下記に該当した（2026-07-29 実測・18点中9 PDF 全部）。

  1. **Type3 フォント**（matplotlib 既定の `pdf.fonttype=3`）
     → Type1/TrueType を要求する誌の自動チェックで弾かれる
  2. **DejaVu Sans**（matplotlib 既定書体）
     → Arial/Helvetica/Times/Courier/Symbol しか許さない誌が多い
  3. **掲載サイズで作っていない**（幅 229 mm など）
     → 90 mm カラムに縮小されると 10 pt が 3.9 pt になり規定違反

このモジュールを図生成スクリプトの冒頭で呼べば、3つとも構造的に起こらない。

使い方
    import sys; sys.path.insert(0, "<repo>/research/scripts")
    from figstyle import use_journal_style, mm, new_figure, savefig, OKABE_ITO

    use_journal_style("elsevier")             # rcParams を投稿仕様に固定
    fig, ax = new_figure("elsevier", "single", height_mm=60)
    ax.plot(x, y, color=OKABE_ITO["blue"], label="control")
    savefig(fig, "figures/fig1")              # fig1.pdf と fig1.png を同時に出す

検証：`python3 figstyle.py` を実行すると、
      ①配色と文字サイズの自己試験（陰性対照・陽性対照つき 12 件）
      ②見本図を作り、**その場で `check_figure.py` に当てて FAIL 数を数える**
      が走る。終了コード 0 = FAIL 0。
      🚨 2026-09-01 まで、この docstring は「FAIL 0 になることを確認済み」と書いていたが、
         実際に当てたことは一度も無く、測ってみると見本図は **FAIL 1**（EM のスタンプ帯に
         軸の枠が入っていた）だった。**「通ったはず」を書かない。判定器を呼んで数える。**
"""

from __future__ import annotations

import os
import re

import matplotlib
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4

# Okabe & Ito の色覚バリアフリー8色（Color Universal Design / jfly）
# Nature Methods (Wong 2011) が紹介して以来、カテゴリ色の事実上の標準。
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
# 🚨🚨 **Okabe-Ito は、この判定器では6色で FAIL・3〜5色でも WARN 止まりになる。**（2026-09-01 実測）
#    check_figure.py の関数（simulate_cvd / delta_e76）で先頭から n 色ずつ測った実測値：
#        2色 minΔE=86.8 PASS
#        3色 minΔE=11.6 WARN（tritanopia: #56B4E9 sky_blue ≈ #009E73 bluish_green）
#        4色 minΔE=11.6 WARN（同上）
#        5色 minΔE=11.6 WARN（同上）
#        6色 minΔE= 6.6 FAIL（deuteranopia: #E69F00 orange ≈ #D55E00 vermillion）
#    判定器の閾値は FAIL が ΔE<8、WARN が ΔE<12。
#    🚨 **「色覚安全と言われているパレット」を名乗ることは、判定を通ることの証明ではない。**
#    → 2色までなら Okabe-Ito のままで PASS。**3色以上で PASS が要るなら下の CVD_SAFE_6 を使う。**

# 白背景で線に使う順（黄色は視認性が低いので後ろ）
OKABE_ITO_CYCLE = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["black"],
    OKABE_ITO["yellow"],
]

# --------------------------------------------------------------------------
# 色覚ゲートを「実測で」通る配色（2026-09-01 追加）
#
# 作り方：Paul Tol (bright / muted / vibrant / high-contrast) と Okabe-Ito を母集団にし、
#   check_figure.py の simulate_cvd / delta_e76 / relative_luminance を目的関数として
#     ・P/D/T いずれのシミュレーション後も全ペアの ΔE76 >= 12（= 色覚 PASS）
#     ・相対輝度の隣接差 >= 0.05（= グレースケール PASS）
#   を満たす組をクリーク探索で全数列挙し、最小ΔE が最大のものを採った。
#   採用値の実測：CVD_SAFE_6 は minΔE=24.8 / 灰階差=0.059。
#   陰性対照：Okabe-Ito の6色は minΔE=6.6（＝この探索の判定が働いていることの確認）。
#   再現＝`python3 figstyle.py --selftest`
# --------------------------------------------------------------------------
CVD_SAFE_6 = {
    "red":    "#EE6677",   # Paul Tol bright
    "green":  "#228833",   # Paul Tol bright
    "cyan":   "#66CCEE",   # Paul Tol bright
    "purple": "#AA3377",   # Paul Tol bright
    "indigo": "#332288",   # Paul Tol muted
    "orange": "#E69F00",   # Okabe-Ito
}
CVD_SAFE_6_LIST = list(CVD_SAFE_6.values())

# 順序尺度（0/1/2 のスコアなど）。輝度が順序を担うので、色覚に依らず順序が読める。
# 実測 minΔE=34.1 / 灰階差=0.381。
SEQ3 = ["#F2F2F2", "#7FB2D9", "#08306B"]
SEQ3_INK = ["#222222", "#0B2E4F", "#FFFFFF"]   # 各段の上に置く文字色


def ink_on(hex_color: str) -> str:
    """塗りの明るさから文字色（白 or 黒）を決める。

    白と黒は check_figure.py の色抽出から外れる（輝度 0.03-0.93 の外）ので、
    何色の上に置いても色覚判定を増やさない。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return "#FFFFFF" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else "#111111"


# check_figure.py の JOURNALS と同期させること
WIDTHS_MM = {
    "nature": {"single": 89.0, "double": 183.0},
    "elsevier": {"single": 90.0, "onehalf": 140.0, "double": 190.0},
    "cellpress": {"single": 85.0, "onehalf": 114.0, "double": 174.0},
    "plos": {"single": 66.8, "double": 190.5},
    "springer": {"single": 84.0, "double": 174.0},
    # JBS(BMC) 逐語 "width of 85 mm ... 170 mm for full page width figure"
    # （check_figure.py の JOURNALS["jbs"] と同じ値。2026-09-07 に同期）
    "jbs": {"single": 85.0, "double": 170.0},
    # ACS: 単段 ≤240pt(84.7mm) / 2段 300-504pt(105.8-177.8mm)。2段には**下限**がある
    "acs": {"single": 84.7, "double": 177.8},
    # T&F は幅を公表していない。値は掲載論文の実測（Crit Rev Anal Chem 2026・US Letter 2段）。
    # ⚠️ 誌が変われば測り直す（FIGURE_GUIDE §7.8b）。T&F は図を段幅に**引き伸ばさない**ので、
    #    小さめは安全・大きめだけが縮小されて pt 規定を割る＝迷ったら小さく作る。
    "tandf": {"single": 89.6, "double": 182.8},
    # Wiley: "Guidelines for the Preparation of Figures" の 80 mm / 180 mm
    # （check_figure.py の JOURNALS["wiley"]["widths_mm"] と同じ値。2026-09-03 に同期）
    "wiley": {"single": 80.0, "double": 180.0},
    # Clinical Chemistry 逐語 "reduced to 1 column (85mm) or 2 columns (176 mm)"
    "clinchem": {"single": 85.0, "double": 176.0},
    # 🚨 CCLM は掲載幅を **公表していない**（Instructions for Authors と Submission
    #    Checklist を独立した2組の語で grep して0件・2026-09-09）。掲載済み PDF も
    #    自動取得できなかった（PMC が拒否）。ここの値は「誌の規定」ではなく既定値で、
    #    clinchem と同じ2段幅を流用している。実物を測れたら置き換える。
    "cclm": {"single": 85.0, "double": 176.0},
    "generic": {"single": 85.0, "double": 174.0},
}
BASE_FONT_PT = {
    "nature": 7.0,
    "elsevier": 8.0,
    "cellpress": 7.5,  # 逐語 "about 6–8 pt at the desired print size"（2026-09-07・旧 9.0 は上限超）
    "plos": 9.0,
    "springer": 9.0,
    "acs": 7.0,      # 誌の下限は 4.5pt。読みやすさを優先して 7pt を基準にする
    "wiley": 8.0,    # 出版社規定なし。generic と同じ 8pt を基準にする
    "clinchem": 8.5, # 誌の規定は「縮小後 8-10 pt」＝下限8を割らないよう8.5から積む
    # CCLM 逐語 "not below 6pt" ＝下限だけで上限が無い。読みやすさを採って 8.5 から積む
    # （下限 6.0 のままだと safe_min_pt が 6.7 になり、実際に組んだときに小さすぎる）。
    "cclm": 8.5,
    "generic": 8.0,
    # JBS(BMC) は pt 下限を書いていない。Springer Nature 全体の lettering 2-3 mm
    # （=8-12 pt）を下限に採り、掲載幅 170 mm で 8 pt を割らないよう 8.8 から積む。
    "jbs": 8.8,
}

# 誌が公表している最小フォント（check_figure.py の JOURNALS['font_pt'][0] と同期）
JOURNAL_MIN_PT = {
    "nature": 5.0, "elsevier": 7.0, "cellpress": 6.0, "plos": 8.0,   # cellpress 6.0 = 逐語 "about 6–8 pt"（2026-09-07 に check_figure と同期・旧 5.0）
    "springer": 7.0, "acs": 4.5, "tandf": 7.0, "generic": 7.0,
    "clinchem": 8.0,   # 逐語 "Letters should be 8-10 points when reduced"
    "cclm": 6.0,       # 逐語 "not below 6pt, preferably in Arial or Helvetica"
    "jbs": 8.0,        # 誌は下限を公表していない。Springer の lettering 2-3 mm を代用
}


def safe_min_pt(journal: str = "generic") -> float:
    """判定器が WARN すら出さない最小 pt。

    🚨 check_figure.py は `下限 <= x < 下限×1.1` を「ぎりぎり」として WARN にする。
       さらに、下限ちょうどを指定すると版面の丸めで下限を下回って FAIL になる（2026-09-01 実測：
       190 mm 版面に 7.0 pt を置いたら「最小 7.0 pt < 下限 7.0 pt」で FAIL）。
       図の中の最小文字は、この値以上にする。"""
    return round(JOURNAL_MIN_PT.get(journal, 7.0) * 1.115, 1)


def pt_scale(journal: str = "generic") -> dict:
    """図の中で使う文字サイズの段階。最小でも safe_min_pt(journal) を下回らない。"""
    lo = safe_min_pt(journal)
    return {
        "tiny":  lo,             # 凡例の補足・細かい目盛
        "small": lo + 0.4,       # 目盛ラベル・凡例
        "body":  lo + 0.8,       # セルの値・注記
        "label": lo + 1.4,       # 行名・列名
        "head":  lo + 2.2,       # 軸タイトル・小見出し
        "big":   lo + 4.2,       # 図番号・強調
    }


def mm(value_mm: float) -> float:
    """mm → inch（matplotlib の figsize 用）"""
    return value_mm / MM_PER_INCH


def _pick_sans() -> list:
    """Arial → Helvetica → 環境にある sans の順で使う。無ければ DejaVu に落ちる。"""
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    order = ["Arial", "Helvetica", "Helvetica Neue", "Liberation Sans", "Nimbus Sans"]
    picked = [name for name in order if name in available]
    return picked + ["DejaVu Sans"]


def use_journal_style(journal: str = "generic", base_pt: float | None = None) -> None:
    """rcParams を投稿仕様に固定する。図を作る前に一度だけ呼ぶ。"""
    pt = base_pt if base_pt else BASE_FONT_PT.get(journal, 8.0)
    # 🚨 mathtext を放置すると、本文が Arial でも `$x^2$` の部分だけ DejaVu で埋め込まれる
    #    （2026-08-04 実測：ACS 図の唯一の書体違反が mathtext 由来だった）。
    #    fontset='custom' にして本文と同じ sans を数式にも使う。
    sans = _pick_sans()
    head = sans[0]
    plt.rcParams.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": head,
            "mathtext.it": f"{head}:italic",
            "mathtext.bf": f"{head}:bold",
            "mathtext.sf": head,
            "mathtext.tt": head,
            "mathtext.cal": f"{head}:italic",
            # ---- 最重要：フォントを TrueType(42) で埋め込む（既定の Type3 を回避）
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # ---- 書体
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            # ---- 文字サイズ（掲載サイズでの実寸 pt）
            "font.size": pt,
            "axes.titlesize": pt,
            "axes.labelsize": pt,
            # 🚨 `pt - 0.5` をそのまま使うと下限を割ることがある（elsevier なら 7.5 < 7.7＝WARN 域）。
            #    safe_min_pt を床にして、目盛・凡例が規定を割らないようにする。
            "xtick.labelsize": max(pt - 0.5, safe_min_pt(journal)),
            "ytick.labelsize": max(pt - 0.5, safe_min_pt(journal)),
            "legend.fontsize": max(pt - 0.5, safe_min_pt(journal)),
            "figure.titlesize": pt + 1,
            # ---- 線・枠（細すぎると印刷で消える：0.5pt を下限に）
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            # ---- 体裁
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO_CYCLE),
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "standard",  # tight は寸法が崩れるので使わない
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def new_figure(
    journal: str = "generic",
    column: str = "single",
    height_mm: float = 60.0,
    **kwargs,
):
    """掲載サイズちょうどの Figure を作る（縮小されない＝フォント規定を割らない）。"""
    width_mm = WIDTHS_MM.get(journal, WIDTHS_MM["generic"])[column]
    return plt.subplots(figsize=(mm(width_mm), mm(height_mm)), **kwargs)


# Editorial Manager（Elsevier / Cell Press）が表示項目の上端に重ね刷りするスタンプの
# 占有域。JMB-D-26-00962_R2.pdf を実測して確定（2026-08-03）。EM は図を拡大縮小せず、
# 組み上がりのページ寸法はアップロードしたファイルと同一なので、この座標は絶対値。
# 詳細と設計規約は FIGURE_GUIDE.md §6.4、機械判定は check_figure.py の「EMスタンプ帯」。
EM_STAMP_Y_MM = (1.49, 8.55)     # 横はページ幅の中央から右端まで
EM_LABEL_MM = (1.76, 1.49, 30.4, 5.02)   # 左上の item 名ラベル x0,y0,x1,y1


def warn_if_em_stamp_collides(pdf_path: str) -> bool:
    """書き出した PDF が EM のスタンプ域に文字・図形を置いていないかを実測する。

    作った瞬間に気づけるのが一番安い。ここでは例外を投げず警告に留める
    （投稿しない図・EM 以外へ出す図もあるため）。投稿前の判定は
    check_figure.py --journal elsevier の「EMスタンプ帯」（FAIL）で行う。
    """
    try:
        import fitz
    except ImportError:
        return False
    to_pt = lambda v: v / 25.4 * 72
    page = fitz.open(pdf_path)[0]
    w, h = page.rect.width, page.rect.height
    stamp = fitz.Rect(w / 2, to_pt(EM_STAMP_Y_MM[0]), w, to_pt(EM_STAMP_Y_MM[1]))
    label = fitz.Rect(*(to_pt(v) for v in EM_LABEL_MM))

    def touches(r):
        if r.height <= 0 or r.width <= 0:
            r = fitz.Rect(r.x0 - .05, r.y0 - .05, r.x1 + .05, r.y1 + .05)
        return r.intersects(stamp) or r.intersects(label)

    hit = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line.get("spans", []):
                if s.get("text", "").strip() and touches(fitz.Rect(s["bbox"])):
                    hit.append(s["text"].strip()[:24])
    for it in page.get_drawings():
        r = it["rect"]
        if r.width >= w * .99 and r.height >= h * .99:
            continue
        if touches(r):
            hit.append("(図形)")
    if hit:
        print(f"  ⚠️ {os.path.basename(pdf_path)}: EM のスタンプ域に {len(hit)} 件"
              f"（{'／'.join(repr(t) for t in hit[:3])}）。Elsevier/Cell Press へ出すなら、"
              f"上端 y {EM_STAMP_Y_MM[1]} mm・ページ幅の中央から右を空けること"
              "（FIGURE_GUIDE.md §6.4）")
    return bool(hit)


def warn_if_text_offpage(fig) -> list:
    """紙の外に落ちた文字を書き出し前に検出する。

    🚨 なぜ必要か（2026-08-04 実測）：軸ラベルが下マージンに収まらないと、matplotlib は
    **警告を出さずに紙の外へ描く**。PDF に痕跡が残らないので `check_figure.py`（文字サイズ・
    書体）も `check_figure_overlap.py`（重なり・はみ出し）も**そもそも見えない**＝両方緑になる。
    「消えた要素」は緑のチェックでは絶対に捕まらないので、作った瞬間にここで捕まえる。
    """
    fig.canvas.draw()
    W, H = fig.bbox.x1, fig.bbox.y1

    # 🚨 誤検出をここで落とす（2026-09-01）。matplotlib は表示範囲の外にも目盛の Text を
    #    作って持っている（xlim 0-10 でも '12' の目盛ラベルが在る）。これは描かれないので
    #    「紙の外に落ちた」ではない。放置すると毎回1件出て、本物の消失を見なくなる。
    ignore = set()
    for ax in fig.get_axes():
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if not (lo - 1e-9 <= tick.get_loc() <= hi + 1e-9):
                    ignore.add(id(tick.label1))
                    ignore.add(id(tick.label2))

    lost = []
    for txt in fig.findobj(match=lambda o: hasattr(o, "get_text")):
        if id(txt) in ignore:
            continue
        s = (txt.get_text() or "").strip()
        if not s or not txt.get_visible():
            continue
        try:
            bb = txt.get_window_extent()
        except Exception:
            continue
        if bb.x1 < 0 or bb.y1 < 0 or bb.x0 > W or bb.y0 > H:
            lost.append(s[:28])
    if lost:
        print(f"  🚨 紙の外に落ちた文字 {len(lost)} 件（{'／'.join(repr(t) for t in lost[:4])}）"
              "＝出力ファイルには存在しない。余白か図の高さを増やすこと")
    return lost


def warn_if_clipped_by_axes(fig, tol_pt: float = 0.05) -> list:
    """軸の枠で切り落とされた図形（模式図の箱・帯・矢印）を、書き出し前に検出する。

    🚨 なぜ必要か（2026-09-07 実測）：`ax.add_patch` で置いた図形は既定で clip_on=True ＝
       **軸の描画領域からはみ出した分は、警告も無しに切られる**。切れるのは紙の内側なので、
       `check_render_residue.py`（紙の端に接するインク）にも `check_figure_overlap.py`
       （要素どうしの重なり）にも **原理的に見えない**。実際、Figure 1 のパネル A で
       最後の箱の右の枠線が、線幅 0.32 mm のうち 0.06 mm しか描かれず、角丸が四角く
       落ちていた。ゲートは 26 本すべて緑のまま、先生が目で見つけられた。

    データの線・点・棒は軸の外で切られるのが正しい（xlim を超えた値は描かない）ので見ない。
    見るのは `ax.patches` と注釈の矢印＝**作図者が「全部見える」つもりで置いた図形**だけ。
    枠線を描く図形は **線幅の半分**を足して測る（経路が境界ちょうどでも、外半分は消えるため）。
    """
    fig.canvas.draw()
    px = fig.dpi / 72.0
    tol = tol_pt * px
    lost = []
    for ax in fig.get_axes():
        cands = list(ax.patches)
        for t in ax.texts:                      # annotate("", ...) の矢印は Text が持つ
            ap = getattr(t, "arrow_patch", None)
            if ap is not None:
                cands.append(ap)
        for art in cands:
            if art is ax.patch or not art.get_visible() or not art.get_clip_on():
                continue
            cb = art.get_clip_box() or ax.bbox
            try:
                bb = art.get_window_extent()
            except Exception:
                continue
            pad = 0.0
            try:
                ec = art.get_edgecolor()
                if art.get_linewidth() > 0 and (len(ec) < 4 or ec[3] > 0):
                    pad = art.get_linewidth() * px / 2.0
            except Exception:
                pass
            over = {"左": cb.x0 - (bb.x0 - pad), "右": (bb.x1 + pad) - cb.x1,
                    "下": cb.y0 - (bb.y0 - pad), "上": (bb.y1 + pad) - cb.y1}
            hit = {k: v for k, v in over.items() if v > tol}
            if hit:
                where = "・".join(f"{k} {v / px / 72 * 25.4:.2f}mm" for k, v in hit.items())
                lost.append(f"{type(art).__name__}({where})")
    if lost:
        print(f"  🚨 軸の枠で切られた図形 {len(lost)} 件（{'／'.join(lost[:4])}）"
              "＝出力では枠線や角が欠ける。座標を軸の内側に収めること")
    return lost

def figure_label(fig, text: str, width_mm: float, height_mm: float,
                 left_mm: float = 2.0, top_mm: float = 6.6,
                 fontsize: float = 12.0) -> None:
    """図番号を、EM の item 名ラベル帯より下の左端に置く。

    🚨 EM は左上 (x 1.76-30.4 mm, y 1.49-5.02 mm) に item 名を、上端 8.55 mm の帯
       （ページ幅の中央から右）にスタンプを焼き込む。図番号を左端の y>=6 mm に置けば
       どちらとも当たらない（check_figure.py の「EMラベル域」「EMスタンプ帯」）。"""
    fig.text(left_mm / width_mm, 1.0 - top_mm / height_mm, text,
             fontsize=fontsize, fontweight="bold", ha="left", va="top",
             color="#111111")


_CMAP_CACHE: dict = {}


def _font_cmap(fp) -> set:
    """FontProperties から、その書体が持つコードポイントの集合を返す。"""
    path = matplotlib.font_manager.findfont(fp)
    if path not in _CMAP_CACHE:
        try:
            from fontTools.ttLib import TTFont
            f = TTFont(path, fontNumber=0, lazy=True)
            cps = set()
            for t in f["cmap"].tables:
                cps |= set(t.cmap.keys())
            _CMAP_CACHE[path] = cps
        except Exception:
            _CMAP_CACHE[path] = None      # 読めない＝判定しない（黙って通さない：下で報告）
    return _CMAP_CACHE[path]


def warn_if_missing_glyph(fig) -> list:
    """書体に無い文字を、書き出し前に見つける。

    🚨 なぜ必要か（2026-09-01 実測）：Arial には ★（U+2605）が無い。matplotlib は
       UserWarning を1行出すだけで、代替字形（豆腐）を描いて PDF を書き出す。
       `check_figure.py` は文字サイズ・書体・色は見るが、**その文字が存在するかは見ない**
       ので緑のまま通る。Arial に無いものの例＝★ ☆ ◆ ▶ ✓ ✔ ※（●■▲ • † ‡ → ≥ ≤ – — は在る）。
    """
    from matplotlib.text import Text
    missing = []
    # 🚨 findobj(hasattr 'get_text') は凡例内部の TextArea にも当たる（get_fontproperties を
    #    持たない）。Text の実体だけを見る。
    for txt in fig.findobj(match=lambda o: isinstance(o, Text)):
        s = txt.get_text() or ""
        if not s.strip() or not txt.get_visible():
            continue
        fp = txt.get_fontproperties()
        cmap = _font_cmap(fp)
        if cmap is None:
            continue
        for ch in set(s):
            if ch in "\n\t ":
                continue
            if ord(ch) not in cmap:
                missing.append((ch, s.strip()[:26]))
    if missing:
        uniq = sorted({c for c, _ in missing})
        print(f"  🚨 書体に無い文字 {len(missing)} 箇所（{' '.join(uniq)}）"
              f"＝PDF には代替字形（豆腐）が入る。例: {missing[0][1]!r}")
    return missing


def warn_if_foreign_font(pdf_path: str) -> list:
    """指定した書体以外が PDF に埋め込まれていないかを実測する。

    🚨 なぜ必要か（2026-09-03 実測）：`use_journal_style` は mathtext の rm/it/bf/sf/tt を
       Arial に差し替えるが、**記号は差し替えられない**。`$\\sim$` や `$\\approx$` を書くと
       Computer Modern（`Cmsy10`）が図に埋め込まれる。
       - `check_figure.py` の書体検査は、誌が書体を規定していないとき（Wiley など）は
         `fonts_allowed=None` で**何も見ない**＝緑のまま通る。
       - PDF からの文字抽出でも化ける（`∼` が `»` として読まれ、重なり判定が誤爆した）。
       対策は「記号に mathtext を使わない」＝Arial にある文字（≈ ≥ ≤ – — × °）を直接書く。
       ⚠️ Arial に **∼ (U+223C) は無い**。`~` か `≈` を使う。
    """
    try:
        import fitz
    except Exception:
        return []
    allowed = {re.sub(r"[^a-z]", "", n.lower()) for n in _pick_sans()}
    foreign = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            for f in page.get_fonts():
                base = re.sub(r"^[A-Z]{6}\+", "", f[3])
                key = re.sub(r"[^a-z]", "", base.lower())
                if not any(key.startswith(a) for a in allowed):
                    foreign.append(base)
    foreign = sorted(set(foreign))
    if foreign:
        print(f"  🚨 {os.path.basename(pdf_path)}: 指定外の書体が埋め込まれている "
              f"{foreign}＝mathtext の記号（$\\sim$ / $\\approx$ 等）が原因のことが多い。"
              f"Arial にある文字を直接書く")
    return foreign


def savefig(fig, stem: str, dpi: int = 600, formats=("pdf", "png")) -> list:
    """PDF（本投稿用ベクタ）と PNG（確認・プレビュー用）を同じ寸法で書き出す。"""
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    warn_if_text_offpage(fig)
    warn_if_missing_glyph(fig)
    written = []
    for ext in formats:
        path = f"{stem}.{ext}"
        fig.savefig(path, format=ext, dpi=dpi)
        written.append(path)
        if ext == "pdf":
            warn_if_em_stamp_collides(path)
            warn_if_foreign_font(path)
    return written


def cvd_safe_pair(name_a: str = "blue", name_b: str = "vermillion") -> tuple:
    """2群比較の既定色。赤緑ではなく青×朱で、P/D/T いずれでも分離する。"""
    return OKABE_ITO[name_a], OKABE_ITO[name_b]


def _selftest(outdir: str = "/tmp/figstyle_selftest") -> str:
    """スタイルが実際に規定を満たす図を作れるかを検証する。

    🚨 2026-09-01：ここが作る見本図は、これまで EM のスタンプ帯（上端 8.55 mm・
       ページ幅の中央から右）に軸の枠が入っており、`check_figure.py --journal elsevier`
       で **FAIL 1** だった。docstring は「FAIL 0 になることを確認済み」と書いていたが、
       実際に当てて確かめてはいなかった。tight_layout は EM 帯を知らないので、
       余白は mm で自分で決める。同時に、y 軸ラベルが紙の外へ落ちてもいた。
    """
    import numpy as np

    W, H = WIDTHS_MM["elsevier"]["single"], 58.0
    use_journal_style("elsevier")
    fig = plt.figure(figsize=(mm(W), mm(H)))
    # 余白は mm 指定。上は EM のスタンプ帯 8.55 mm を必ず空ける。
    left, right, top, bottom = 13.0, 2.0, 10.5, 11.0
    ax = fig.add_axes([left / W, bottom / H,
                       (W - left - right) / W, (H - top - bottom) / H])
    x = np.linspace(0, 10, 100)
    for i, (label, shift) in enumerate([("control", 0.0), ("treated", 0.6)]):
        ax.plot(x, np.sin(x) + shift, label=label, linestyle=["-", "--"][i])
    ax.set_xlabel("time (h)")
    ax.set_ylabel("relative level")
    ax.legend(loc="upper right")
    stem = os.path.join(outdir, "selftest")
    paths = savefig(fig, stem)
    plt.close(fig)
    return paths[0]


def _selftest_against_gate(pdf_path: str) -> int:
    """見本図を check_figure.py に実際に当てて、FAIL 0 を数える。

    🚨 「FAIL 0 のはず」と書くのではなく、判定器を呼んで数える。
       判定器が読めない環境（PyMuPDF 無し等）では 0 を返さず、その旨を出す。"""
    import subprocess
    import sys as _sys
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([_sys.executable, os.path.join(here, "check_figure.py"),
                        pdf_path, "--journal", "elsevier", "--column", "single"],
                       capture_output=True, text=True)
    m = re.search(r"合計: FAIL (\d+) 件 / WARN (\d+) 件", r.stdout)
    if not m:
        print("  ⚠️ check_figure.py の出力を読み取れなかった（判定していない）")
        return 1
    n_fail, n_warn = int(m.group(1)), int(m.group(2))
    print(f"  見本図を判定器に当てた結果: FAIL {n_fail} 件 / WARN {n_warn} 件")
    if n_fail:
        for line in r.stdout.splitlines():
            if "[FAIL]" in line:
                print("   ", line.strip())
    return 1 if n_fail else 0


def selftest_palettes() -> int:
    """配色と文字サイズの規定を、判定器そのものの関数で検証する（陰性対照つき）。

    🚨 ここで測るのは「そう決めたから正しい」ではなく「判定器が通すか」。
       陰性対照（Okabe-Ito 6色・下限ちょうどの pt）が**落ちること**まで確認する。
       落ちない対照を置いた検査は、何も見ていない。
    """
    import itertools
    import sys as _sys

    import numpy as np

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_figure import (simulate_cvd, srgb_to_lab, delta_e76,
                              relative_luminance)

    def _rgb(h):
        h = h.lstrip("#")
        return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

    def worst(hexes):
        cols = np.array([_rgb(h) for h in hexes])
        w = 1e9
        for kind in ("protanopia", "deuteranopia", "tritanopia"):
            lab = srgb_to_lab(simulate_cvd(cols, kind))
            for i, j in itertools.combinations(range(len(cols)), 2):
                w = min(w, delta_e76(lab[i], lab[j]))
        return w

    def gap(hexes):
        lums = sorted(relative_luminance(np.array(_rgb(h))) for h in hexes)
        return min(b - a for a, b in zip(lums, lums[1:]))

    fails, checks = [], 0

    # [1] CVD_SAFE_6 が色覚 PASS の域（ΔE>=12）にある
    checks += 1
    if worst(CVD_SAFE_6_LIST) < 12:
        fails.append(f"CVD_SAFE_6 の最小ΔE {worst(CVD_SAFE_6_LIST):.1f} < 12")
    # [2] CVD_SAFE_6 がグレースケール PASS の域（隣接輝度差>=0.05）にある
    checks += 1
    if gap(CVD_SAFE_6_LIST) < 0.05:
        fails.append(f"CVD_SAFE_6 の灰階差 {gap(CVD_SAFE_6_LIST):.3f} < 0.05")
    # [3] SEQ3（順序尺度）が色覚 PASS の域にある
    checks += 1
    if worst(SEQ3) < 12:
        fails.append(f"SEQ3 の最小ΔE {worst(SEQ3):.1f} < 12")
    # [4] 陰性対照＝Okabe-Ito の6色は落ちる（落ちなければ [1] は何も見ていない）
    checks += 1
    okabe6 = [OKABE_ITO[k] for k in
              ("orange", "sky_blue", "bluish_green", "yellow", "blue", "vermillion")]
    if worst(okabe6) >= 12:
        fails.append(f"陰性対照が落ちない＝色覚の判定が働いていない"
                     f"（Okabe-Ito 6色 minΔE={worst(okabe6):.1f}）")
    # [5] 対照＝Okabe-Ito の5色は FAIL 域（<8）には落ちない。
    #     これが落ちるなら判定が厳しすぎて何でも弾いていることになる。
    #     🚨 ただし 5色でも minΔE=11.6 で WARN 域（<12）には入る＝**PASS ではない**。
    #        「Okabe-Ito なら安全」で済ませられないことの根拠がこの1行。
    checks += 1
    if worst(okabe6[:5]) < 8:
        fails.append(f"Okabe-Ito 5色が FAIL 域＝判定が厳しすぎる"
                     f"（minΔE={worst(okabe6[:5]):.1f}）")
    checks += 1
    if worst(okabe6[:5]) >= 12:
        fails.append(f"Okabe-Ito 5色が PASS 域に入った＝実測(11.6)と違う。"
                     f"判定器が変わった可能性があるので CVD_SAFE_6 の根拠を測り直すこと")
    # [6] pt_scale の最小が safe_min_pt 以上
    checks += 1
    for j in ("elsevier", "nature", "springer", "generic"):
        sc = pt_scale(j)
        if min(sc.values()) < safe_min_pt(j):
            fails.append(f"{j}: pt_scale の最小 {min(sc.values())} < {safe_min_pt(j)}")
    # [7] 陰性対照＝下限ちょうどは safe_min_pt を下回る（＝判定が WARN 域を見ている）
    checks += 1
    if JOURNAL_MIN_PT["elsevier"] >= safe_min_pt("elsevier"):
        fails.append("safe_min_pt が下限と同じ＝WARN 域を避けられていない")
    # [8] ink_on が白と黒を返し分ける
    checks += 1
    if ink_on("#332288") != "#FFFFFF" or ink_on("#F0E442") != "#111111":
        fails.append("ink_on の白黒判定が反転している")
    # [9] Arial が実在する（無ければ DejaVu に落ちて書体 WARN になる）
    checks += 1
    if "Arial" not in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        fails.append("Arial が見つからない＝DejaVu に落ちる")

    print(f"  CVD_SAFE_6 minΔE={worst(CVD_SAFE_6_LIST):5.1f} 灰階差={gap(CVD_SAFE_6_LIST):.3f}")
    print(f"  SEQ3       minΔE={worst(SEQ3):5.1f} 灰階差={gap(SEQ3):.3f}")
    print(f"  陰性対照 Okabe-Ito 6色 minΔE={worst(okabe6):5.1f}（12 未満で正常）")
    print(f"  対照 Okabe-Ito 5色     minΔE={worst(okabe6[:5]):5.1f}（8 以上 12 未満で正常＝WARN 域）")
    for f in fails:
        print("  FAIL", f)
    # [11] 陽性対照＝本当に紙の外へ出た文字は warn_if_text_offpage が拾う
    checks += 1
    fig = plt.figure(figsize=(mm(60), mm(40)))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_ylabel("a very long axis label that cannot fit in this margin at all")
    if not warn_if_text_offpage(fig):
        fails.append("陽性対照が拾われない＝warn_if_text_offpage が何も見ていない")
    plt.close(fig)
    # [12] 陰性対照＝表示範囲外の目盛ラベルは「消失」に数えない
    checks += 1
    fig = plt.figure(figsize=(mm(80), mm(50)))
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.72])
    ax.plot([0, 10], [0, 1])
    ax.set_xlim(0, 10)
    if warn_if_text_offpage(fig):
        fails.append("表示範囲外の目盛ラベルを誤検出している")
    plt.close(fig)

    # [13] 陽性対照＝Arial に無い ★ を置くと warn_if_missing_glyph が拾う
    #     🚨 先に use_journal_style を呼ぶこと。呼ばないと matplotlib 既定の DejaVu Sans が
    #        使われ、DejaVu には ★ が在るので対照が成立しない（2026-09-01 に実際に空振りした）。
    use_journal_style("elsevier")
    checks += 1
    fig = plt.figure(figsize=(mm(60), mm(30)))
    fig.text(0.5, 0.5, "\u2605 star", fontsize=10)
    if not warn_if_missing_glyph(fig):
        fails.append("陽性対照が拾われない＝warn_if_missing_glyph が何も見ていない")
    plt.close(fig)
    # [14] 陰性対照＝Arial に在る文字だけなら鳴らない
    checks += 1
    fig = plt.figure(figsize=(mm(60), mm(30)))
    fig.text(0.5, 0.5, "\u2022 \u2192 \u2265 \u2013 ok", fontsize=10)
    if warn_if_missing_glyph(fig):
        fails.append("在る文字を『無い』と言っている＝誤検出")
    plt.close(fig)

    # [15] 陽性対照＝軸（xlim 0-100）の外へ出た箱は拾われる（2026-09-07 Figure 1 パネル A の形）
    checks += 1
    from matplotlib.patches import Rectangle as _Rect
    fig = plt.figure(figsize=(mm(60), mm(30)))
    ax = fig.add_subplot(111); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    ax.add_patch(_Rect((90, 2), 11, 6, fc="none", ec="0.3", lw=0.9))   # 右へ 1 単位はみ出す
    if not warn_if_clipped_by_axes(fig):
        fails.append("陽性対照が拾われない＝warn_if_clipped_by_axes が何も見ていない")
    plt.close(fig)
    # [16] 陰性対照は本番で当たる 3 通りをまとめて置く（1 標本の対照は対照ではない）：
    #      ①軸の内側の箱 ②軸の縁ちょうどから始まる棒（barh は既定で枠線を描かない）
    #      ③xlim の外まで伸びるデータの線＝**切られるのが正しい**
    checks += 1
    fig = plt.figure(figsize=(mm(60), mm(30)))
    ax = fig.add_subplot(111); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    ax.add_patch(_Rect((10, 2), 20, 6, fc="none", ec="0.3", lw=0.9))
    ax.barh([5], [40], left=0, height=1.0, color="0.6")
    ax.plot([-50, 150], [8, 8], lw=0.8)
    if warn_if_clipped_by_axes(fig):
        fails.append("切れていないものを『切れた』と言っている＝誤検出")
    plt.close(fig)

    print(f"配色・文字サイズの自己試験: {checks - len(fails)}/{checks}")
    return 1 if fails else 0


if __name__ == "__main__":
    rc = selftest_palettes()
    pdf = _selftest()
    print("wrote:", pdf)
    rc |= _selftest_against_gate(pdf)
    print(f"figstyle selftest: {'PASS' if rc == 0 else 'FAIL'}")
    raise SystemExit(rc)
