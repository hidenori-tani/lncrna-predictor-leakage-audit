#!/usr/bin/env python
"""Assemble the R2 text from its sources: no number in the output is typed by hand.

  manuscript_r2.src.md  + blocks_r2.md + references_r2.md + key_numbers_r2.json  ->  manuscript_r2.md
  response_to_reviewers_r2.src.md + blocks_r2.md + key_numbers_r2.json          ->  response_to_reviewers_r2.md
  tables_r2.src.md (from make_numbers_r2.py) + blocks_r2.md                      ->  tables_r2.md (Table 9 filled)

<<NAME>> is replaced by the block "=== NAME ===" of blocks_r2.md; {{key}} by key_numbers_r2.json's
formatted value. In the response letter, <<QUOTE:phrase>> becomes the manuscript paragraph that
contains the phrase (<<QUOTE:Section|phrase>>: within that section), and <<QUOTESENT:Section|phrase>> the manuscript sentence that contains the phrase
within the section whose heading contains "Section", labelled with that heading; both are taken from
the assembled manuscript, so every passage the letter quotes is the manuscript's own text, character
for character. The run fails if any marker survives, if a block is unused, if a quoted phrase is not
found exactly once, or if the numbers file was written in --partial mode.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABBREV = ("et al.", "Fig.", "Figs.", "e.g.", "i.e.", "cf.", "vs.", "No.", "Eq.")


def blocks():
    text = (HERE / "blocks_r2.md").read_text(encoding="utf-8")
    out, name, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^=== ([A-Z0-9_]+) ===\s*$", line)
        if m:
            if name:
                out[name] = "\n".join(buf).strip()
            name, buf = m.group(1), []
        elif name:
            buf.append(line)
    if name:
        out[name] = "\n".join(buf).strip()
    return out


def sections(ms_text):
    """[(heading, paragraphs)] in document order; the title counts as a heading."""
    out, head, paras = [], "", []
    for para in ms_text.split("\n\n"):
        s = para.strip()
        if not s:
            continue
        if s.startswith("#"):
            out.append((head, paras))
            head, paras = s.lstrip("#").strip(), []
        else:
            paras.append(s)
    out.append((head, paras))
    return out


def sentence_around(para, phrase):
    i = para.index(phrase)

    def is_end(k):        # para[k] is '.', '?' or '!' and ends a sentence
        if k + 1 < len(para) and not para[k + 1].isspace():
            return False
        j = k + 1
        while j < len(para) and para[j].isspace():
            j += 1
        if j < len(para) and not (para[j].isupper() or para[j] in "(\"“'"):
            return False
        return not any(para[:k + 1].endswith(a) for a in ABBREV)

    a = i
    while a > 0 and not (para[a - 1] in ".?!" and is_end(a - 1)):
        a -= 1
    b = i + len(phrase)
    while b < len(para) and not (para[b] in ".?!" and is_end(b)):
        b += 1
    return para[a:b + 1].strip()


def quote(text, ms_text, where):
    """<<QUOTE:phrase>> -> the manuscript paragraph containing the phrase, as a block quote;
    <<QUOTESENT:Section|phrase>> -> the sentence containing it within that section, labelled."""
    paras = [q for q in ms_text.split("\n\n") if q.strip() and not q.lstrip().startswith("#")]
    secs = sections(ms_text)

    def rq(m):
        sec, _, phrase = m.group(1).rpartition("|")
        if sec:          # <<QUOTE:Section|phrase>>: the paragraph within that section
            hits = [p for h, ps in secs if sec.strip() in h for p in ps if phrase in p]
        else:
            hits = [q for q in paras if phrase in q]
        if len(hits) != 1:
            sys.exit(f"{where}: quote phrase found in {len(hits)} manuscript paragraphs: {phrase[:60]!r}")
        return "\n".join("> " + line for line in hits[0].strip().splitlines())

    def rs(m):
        sec, phrase = m.group(1).strip(), m.group(2).strip()
        cand = [(h, p) for h, ps in secs if sec in h for p in ps if phrase in p]
        if len(cand) != 1:
            sys.exit(f"{where}: sentence phrase found {len(cand)} times in sections matching {sec!r}: {phrase[:60]!r}")
        h, p = cand[0]
        label = h if h != "Abstract" else "Abstract"
        return f"> {label}: {sentence_around(p, phrase)}"

    text = re.sub(r"<<QUOTE:(.+?)>>", rq, text)
    return re.sub(r"<<QUOTESENT:([^|>]+)\|(.+?)>>", rs, text)


def fill_blocks(text, B, used, where):
    for _ in range(3):                     # blocks may carry blocks
        def rb(m):
            k = m.group(1)
            if k not in B:
                sys.exit(f"{where}: block <<{k}>> is not defined in blocks_r2.md")
            used.add(k)
            return B[k]
        text = re.sub(r"<<([A-Z0-9_]+)>>", rb, text)
    return text


def fill_keys(text, F, where):
    def rk(m):
        k = m.group(1)
        if k not in F:
            sys.exit(f"{where}: number {{{{{k}}}}} is not in key_numbers_r2.json")
        return F[k]
    text = re.sub(r"\{\{([a-z0-9_]+)\}\}", rk, text)
    left = re.findall(r"<<[^>\n]*>>|\{\{[^}\n]*\}\}", text)
    if left:
        sys.exit(f"{where}: unresolved markers {left[:5]}")
    return text


def main():
    kn = json.loads((HERE / "key_numbers_r2.json").read_text(encoding="utf-8"))
    if kn.get("partial") and "--allow-partial" not in sys.argv:
        sys.exit("key_numbers_r2.json was written with --partial: some results are missing")
    F = kn["fmt"]
    B = blocks()
    B.setdefault("REFERENCES", (HERE / "references_r2.md").read_text(encoding="utf-8").strip())
    used = set()
    jobs = [("manuscript_r2.src.md", "manuscript_r2.md"),
            ("response_to_reviewers_r2.src.md", "response_to_reviewers_r2.md"),
            ("tables_r2.src.md", "tables_r2.md")]
    ms_text = None
    for src, dst in jobs:
        t = fill_blocks((HERE / src).read_text(encoding="utf-8"), B, used, src)
        t = fill_keys(t, F, src) if ms_text is None else fill_keys(quote(t, ms_text, src), F, src)
        if dst == "manuscript_r2.md":
            ms_text = t
        (HERE / dst).write_text(t, encoding="utf-8")
        print(f"  ✔ {dst}")
    unused = sorted(set(B) - used - {"REFERENCES"})
    if unused:
        sys.exit(f"blocks defined but never used: {unused}")


if __name__ == "__main__":
    main()
