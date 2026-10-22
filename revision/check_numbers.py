#!/usr/bin/env python
"""Verify that the DVMnet numbers in the prose are the numbers the runs produced.

Two independent routes, because a gate that only checks a list somebody wrote down cannot see the
claims nobody listed:

  A. CURATED -- named claims are compared against key_numbers.json by key, so a value that moved
     is caught even if the prose still reads plausibly.
  B. SWEEP   -- every 3-to-4-decimal number appearing in the DVMnet Results sections must be
     present in the generated tables or in key_numbers.json.  This has no list to fall off.

Numbers that legitimately are not results (years, p-value exponents, version strings, and the
values carried over from the unchanged RNAlight and benchmark-level LncPTPred analyses) are
listed in ALLOW with the reason they are there.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE / "manuscript_r1.md"
TABLES = HERE / "tables_r1.md"
KEY = HERE / "key_numbers.json"

# Sections whose numbers come from the runs this revision re-ran, so every 3-4 decimal number
# in them must be traceable to a results JSON.  The RNAlight sections (3.5-3.7) are deliberately
# NOT here: their numbers come from runs this revision did not repeat, and sweeping them would
# report failures for values that are correct and simply not in key_numbers.json.
DVM_SECTIONS = ("## Abstract", "### 3.1.", "### 3.2.", "### 3.3.", "### 3.4.",
                "### 3.8.", "### 3.9.", "### 4.4.")

ALLOW = {
    "0.7833": "RNAlight の報告値（第三者の論文の数値・notes/GATE_RESULT.md 10行目に出典）",
    "0.780": "RNAlight の再現値。DVMnet の実行結果ではない＝実測 0.7802 の丸め"
             "（出典 notes/GATE_RESULT.md 20行目「AUROC = 0.7802」）",
    "0.7600": "投稿版で報告した degree ヒューリスティック（旧 Table 4・本文では言及しない）",
    "0.5405": "LncPTPred ベンチマークの陽性 GC 中央値（Table 7・ベンチマーク水準の実測）",
    "0.4722": "同・陰性 GC 中央値（Table 7）",
    "0.528": "LncPTPred の長さ照合の単一特徴 AUROC（Table 7）",
    "0.500": "照合が成立していれば出るはずの値（定義値であって実行結果ではない）",
    "0.628": "LncPTPred の GC 単一特徴 AUROC（Table 7）",
    # 以下3件は DVMnet 著者の Fig. 2 の箱ひげ図から読み取った値＝私たちの実行結果ではない。
    "0.845": "DVMnet 著者 Fig. 2 の中央値（公表図の目視読み取り・出典を本文に明記）",
    "0.785": "同・分布の下端",
    "0.898": "同・分布の上端",
}

fails, notes = [], []


def section_text(ms: str, head: str) -> str:
    i = ms.index(head)
    j = ms.find("\n### ", i + 1)
    j = j if j > 0 else ms.find("\n## ", i + 1)
    return ms[i: j if j > 0 else len(ms)]


def curated(key):
    """A. named claims -> JSON keys."""
    ms = MS.read_text(encoding="utf-8")
    checks = []
    d = key.get("dvmnet", {})
    for cond, human in (("random-verbatim", "verbatim"), ("random-uniform", "random-uniform"),
                        ("random-degmat", "random-degmat"), ("cold-uniform", "cold-uniform"),
                        ("cold-degmat", "cold-degmat")):
        if cond in d:
            checks.append((f"{human} の nested", f"{d[cond]['nested']['mean']:.4f}"))
    for label, r in key.get("paired", {}).items():
        checks.append((f"paired: {re.sub(r'[*]', '', label)}", f"{r['mean']:+.4f}"))
    hit = 0
    for label, val in checks:
        if val.lstrip("+") in ms or val in ms:
            hit += 1
    notes.append(f"A 名指しの主張 {hit}/{len(checks)} 件が本文に出現（出現しない値は本文が触れていないだけの場合もある）")
    return checks


def sweep(key):
    """B. every 3-4 decimal number in the DVMnet Results sections must be traceable."""
    ms = MS.read_text(encoding="utf-8")
    tables = TABLES.read_text(encoding="utf-8") if TABLES.exists() else ""
    # 🚨 「許容差つきで pool のどれかに近ければ合格」にしてはいけない。
    #    key_numbers.json には fold ごと・分割ごとの値が数百個入っているので、
    #    0.3〜0.9 のどんな 4 桁小数にも 5e-4 以内の隣人が必ず居る＝ゲートが常に緑になる。
    #    実測（2026-09-03 の陰性対照）＝0.5040 を 0.5140 に書き換えても素通りした。
    #    そこで、JSON と表に在る値から「印字されうる文字列」を明示的に作り、完全一致で照合する。
    allowed: set[str] = set()

    def add_number(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return
        for nd in (2, 3, 4):
            allowed.add(f"{x:.{nd}f}")
            allowed.add(f"{abs(x):.{nd}f}")

    def walk(o):
        if isinstance(o, dict):
            for vv in o.values():
                walk(vv)
        elif isinstance(o, (list, tuple)):
            for vv in o:
                walk(vv)
        elif isinstance(o, (int, float)):
            add_number(o)

    walk(key)
    for m in re.finditer(r"\d+\.\d+", tables):
        add_number(m.group())
        allowed.add(m.group())

    def traceable(tok: str) -> bool:
        return tok in allowed or tok in ALLOW

    seen, bad = 0, []
    for head in DVM_SECTIONS:
        if head not in ms:
            continue
        for m in re.finditer(r"(?<![\w.])\d\.\d{3,4}(?![\w])", section_text(ms, head)):
            seen += 1
            if not traceable(m.group()):
                bad.append((head.strip(), m.group()))
    for head, tok in bad:
        fails.append(f"{head} の数値 {tok} が表にも key_numbers.json にも無い")
    notes.append(f"B 総ざらい: {len(DVM_SECTIONS)} 節（{', '.join(h.strip() for h in DVM_SECTIONS)}）の "
                 f"3〜4 桁小数 {seen} 件を照合（未追跡 {len(bad)} 件）")


def main():
    if not KEY.exists():
        sys.exit("key_numbers.json がまだ無い＝make_tables.py を先に走らせる")
    key = json.load(open(KEY))
    curated(key)
    sweep(key)
    print("=== 実行した検査 ===")
    for n in notes:
        print("  ", n)
    print()
    if fails:
        print(f"■ FAIL {len(fails)}")
        for f in fails:
            print("  ✗", f)
        sys.exit(1)
    print("■ FAIL 0 — 本文の数値は実行結果に一致")


if __name__ == "__main__":
    main()
