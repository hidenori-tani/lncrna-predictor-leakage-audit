#!/usr/bin/env python
"""Study-effort proxy: PubMed co-occurrence count per lncRNA gene symbol.
Public GET to NCBI E-utilities only (no data uploaded). Resumable cache.
Rate-limited to <=3 req/s (no API key).
"""
import sys, time, json, urllib.parse, subprocess, os
import pandas as pd

# --- path config: edit these or set env vars. Defaults assume repo layout code/ + data/ ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
DATA = _os.environ.get('AUDIT_DATA', _os.path.join(_ROOT, 'data'))
FIG  = _os.environ.get('AUDIT_FIGS', _os.path.join(_ROOT, 'figures'))
RNALIGHT_DIR = _os.environ.get('RNALIGHT_DIR', _os.path.join(_ROOT, 'external', 'RNAlight'))
DVMNET_DIR   = _os.environ.get('DVMNET_DIR',   _os.path.join(_ROOT, 'external', 'DVMnet'))
# --- end path config ---
REPO = RNALIGHT_DIR
BASE = f"{REPO}/lncRNA/03_Model_Construction"
OUT  = _os.path.join(DATA, "pubmed_counts.tsv")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
LNC = '("lncRNA" OR "long noncoding RNA" OR "long non-coding RNA" OR "lincRNA")'

tr = pd.read_csv(f"{BASE}/lncRNA_sublocation_TrainingSet.tsv", sep="\t")
te = pd.read_csv(f"{BASE}/lncRNA_sublocation_TestSet.tsv", sep="\t")
pool = pd.concat([tr, te], ignore_index=True).drop_duplicates("ensembl_transcript_id")
symbols = sorted(set(pool["name"].dropna().astype(str)))
print(f"unique gene symbols to query: {len(symbols)}", flush=True)

done = {}
if os.path.exists(OUT):
    for line in open(OUT):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2: done[p[0]] = p[1]
    print(f"resuming: {len(done)} already cached", flush=True)

def count(sym):
    # use curl (system cert store works in this env where python urllib SSL fails)
    term = f'"{sym}" AND {LNC}'
    url = EUTILS + "?" + urllib.parse.urlencode({"db":"pubmed","term":term,"rettype":"count","retmode":"json"})
    out = subprocess.run(["curl","-s","--max-time","30",url], capture_output=True, text=True, timeout=40).stdout
    d = json.loads(out)
    return int(d["esearchresult"]["count"])

f = open(OUT, "a")
n=0
for sym in symbols:
    if sym in done: continue
    try:
        c = count(sym)
    except Exception as e:
        c = -1  # mark errors, retry later
        sys.stderr.write(f"err {sym}: {e}\n")
    f.write(f"{sym}\t{c}\n"); f.flush()
    n+=1
    if n % 200 == 0: print(f"  {n} queried (last {sym}={c})", flush=True)
    time.sleep(0.34)  # ~3/s
f.close()
print(f"DONE: queried {n} new symbols; cache -> {OUT}", flush=True)
