# What lncRNA predictors actually learn — audit code

Reproduction code for:

> Tani H. *What lncRNA predictors actually learn: an audit of confounding and
> evaluation leakage, with a reporting standard.* Computers in Biology and Medicine (submitted, 2026).

This repository contains the complete audit pipeline for two published lncRNA
machine-learning predictors — **RNAlight** (nuclear/cytoplasmic localization,
k-mer LightGBM) and **DVMnet** (lncRNA–miRNA interaction, graph model). All
experiments run on CPU (no GPU required).

## 1. Data sources

This study analyzes only publicly available data. Obtain the primary inputs from:

| Predictor | Upstream repository | What we use |
|---|---|---|
| RNAlight | https://github.com/YangLab/RNAlight | processed train/test lncRNA sets + released k-mer frequency matrix (featurization check only) |
| DVMnet | https://github.com/liuliwei1980/DVMnet | lncRNA–miRNA edge list + node table |

`data/` here ships the small **derived** artifacts so the expensive steps can be
skipped:
- `pool.fasta` — 3,792 de-duplicated pooled lncRNA sequences (from the RNAlight release)
- `pool_features.npz` — precomputed 1,344-dim k-mer frequency matrix + labels
- `mmseqs_clusters/lin_0{5,7,8,9}_cluster.tsv`, `lin_80_cluster.tsv` — MMseqs2 clusters at 0.5/0.7/0.8/0.9 identity
- `pubmed_counts.tsv` — PubMed co-occurrence counts (study-effort proxy) for 3,765/3,792 loci

## 2. Environment

```bash
# option A — conda (includes MMseqs2)
conda env create -f environment.yml && conda activate lncrna-leakage-audit
# option B — pip (install MMseqs2 separately: `brew install mmseqs2`)
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```
Python 3.13; MMseqs2 release 18-8cc5c. Seeds are fixed (100 for RNAlight to match
the original; 42 for DVMnet).

### Paths (no hard-coded paths — set via env vars, with sensible defaults)
Each script resolves paths relative to the repo (`code/` + `data/`), overridable by
environment variables:
```bash
export RNALIGHT_DIR=/path/to/your/clone/RNAlight   # default: ../external/RNAlight
export DVMNET_DIR=/path/to/your/clone/DVMnet       # default: ../external/DVMnet
# AUDIT_DATA (default ../data) and AUDIT_FIGS (default ../figures) rarely need changing
```
Scripts that use only the bundled `data/` run out of the box; scripts that read the
upstream train/test tables or the DVMnet edge list need the corresponding clone (see
the "needs" column below).

## 3. Scripts → reported results

| Script | Produces | Needs | Manuscript |
|---|---|---|---|
| `prep_pool.py` | `pool.fasta`, `pool_features.npz` from the RNAlight release | RNALIGHT_DIR | §2.1 (3,792 unique lncRNAs) |
| `reproduce_rnalight.py` | featurization check (max\|Δ\| = 9.97e-17) + reproduced AUROC 0.780 vs 0.783 | RNALIGHT_DIR | §2.2, §3.1 |
| `exp_homology_split.py` | random CV 0.728; MMseqs2 GroupKFold 0.728 (id0.5) / 0.725 (id0.8); homolog coverage 1.6–5.2% | bundled `data/` ✓ | §3.1, Fig 1A |
| `exp_composition_confounds.py` | AUROC by feature set (dinuc-16 = 0.703; length p=3.9e-24, GC p=4.6e-13) | RNALIGHT_DIR | §3.2, Fig 1B |
| `exp_confound_matched.py` | length/GC-matched AUROC 0.689 (n=2,764); 8-seed CV 0.730 ± 0.004 | RNALIGHT_DIR | §3.2 |
| `crawl_pubmed_counts.py` | `pubmed_counts.tsv` (resumable; curl-based) | RNALIGHT_DIR | §2.5 |
| `exp_study_effort.py` | strata AUROC 0.730/0.719/0.672/0.716; studied→unstudied 0.690 vs 0.714; 75% zero-pub | RNALIGHT_DIR + bundled `data/pubmed_counts.tsv` | §3.3 |
| `exp_dvmnet_topology.py` | degree-sum 0.750 (their protocol); all heuristics 0.500 cold-start | DVMNET_DIR | §3.4, Fig 2 |
| `exp_dvmnet_negatives.py` | degree-matched negatives 0.458 (canonical) | DVMNET_DIR | §3.4, Fig 2A |
| `make_figures.py` | `figures/fig1_rnalight.{png,pdf}`, `fig2_dvmnet.{png,pdf}` | (self-contained) | Fig 1, 2 |

## 4. Quick reproduce

Runs out of the box on the bundled `data/` (no upstream clone needed):
```bash
cd code
python exp_homology_split.py          # Fig 1A numbers (verified: random 0.728, homology 0.728/0.725)
python make_figures.py                # regenerate both figures from recorded values
```
The remaining scripts additionally need the upstream clones — set `RNALIGHT_DIR` /
`DVMNET_DIR` (see §2), then:
```bash
python exp_composition_confounds.py   # Fig 1B numbers        (RNALIGHT_DIR)
python exp_confound_matched.py        # matched + multi-seed  (RNALIGHT_DIR)
python exp_study_effort.py            # study-effort strata   (RNALIGHT_DIR)
python exp_dvmnet_topology.py         # DVMnet topology leak  (DVMNET_DIR)
python exp_dvmnet_negatives.py        # degree-matched negs   (DVMNET_DIR)
```

## 5. License

Audit code: MIT (see `LICENSE`). Upstream data retains its original license.
