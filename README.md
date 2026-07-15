# Protocol ablation — audit code for three lncRNA machine-learning predictors

Reproduction code for:

> Tani H. *Protocol Ablation: Attributing the Reported Performance of lncRNA
> Machine-Learning Predictors to Evaluation Artifacts.* (submitted, 2026).

A reported benchmark score is a joint property of a model and its evaluation protocol.
**Protocol ablation** re-runs the published model itself under a factorial ladder of
counterfactual protocols — each closing exactly one defect — so the reported score can be
*attributed* rather than merely doubted.

Three audited predictors, one collapse and two non-collapses:

| Predictor | Task | Result under the ladder |
|---|---|---|
| **RNAlight** (Yuan et al. 2023) | nuclear/cytoplasmic localization | does **not** collapse; homology-aware splitting is inert (ΔAUROC < 0.01) |
| **DVMnet** (Wei et al. 2025) | lncRNA–miRNA interaction | **collapses**: 0.803 → 0.510 (chance) when only the negatives are fixed |
| **LncPTPred** (Das et al. 2025) | lncRNA–protein interaction | does **not** collapse; GC-matching costs only 0.031 |

Everything runs on **CPU** (no GPU).

## 1. Data sources — we redistribute none of the audited data

This study analyzes only publicly available data, obtained from the original authors:

| Predictor | Upstream repository |
|---|---|
| RNAlight | https://github.com/YangLab/RNAlight |
| DVMnet | https://github.com/liuliwei1980/DVMnet |
| LncPTPred | https://github.com/zglabDIB/lncptpred |

The audit is only meaningful when run against the authors' own released artifacts, so this
repository ships **derived** artifacts only:

- `data/pool.fasta`, `data/pool_features.npz` — pooled lncRNA sequences + k-mer matrix (RNAlight)
- `data/mmseqs_clusters/*.tsv` — MMseqs2 clusters at 0.5/0.7/0.8/0.9 identity
- `data/pubmed_counts.tsv` — PubMed co-occurrence counts (study-effort proxy)
- `data/dvmnet_embed_seed42.npz` — **cached doc2vec node embeddings** for DVMnet

The embedding cache matters: gensim's doc2vec is not deterministic, so without it the
protocol effects would be confounded with embedding noise. Shipping it makes our ablation
reproducible *exactly* rather than up to that noise.

## 2. Environment

```bash
conda env create -f environment.yml && conda activate lncrna-leakage-audit   # includes MMseqs2
# or
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```
Python 3.13.5; torch 2.13.0, torch_geometric 2.8.0, gensim 4.4.0, LightGBM 4.6,
scikit-learn 1.9, MMseqs2 18, ViennaRNA 2.7.2. Seeds fixed (100 for RNAlight to match the
original; 42 for DVMnet).

### Paths
No hard-coded paths. Scripts resolve relative to the repo, overridable by env var:
```bash
export RNALIGHT_DIR=/path/to/RNAlight     # default ../external/RNAlight
export DVMNET_DIR=/path/to/DVMnet         # default ../external/DVMnet
# AUDIT_DATA (../data), AUDIT_RESULTS (../results), AUDIT_FIGS (../figures) rarely need changing
```

### Running DVMnet's model on CPU
`code/dvmnet_harness.py` leaves the upstream repository **unmodified**. Two shims make its
CUDA-only code run on CPU without touching any model math: `torch.Tensor.cuda(...)` becomes
the identity (the model calls `.cuda(0)` on already-CPU tensors) and `torch.cuda.set_device(...)`
becomes a no-op. Architecture, hyperparameters (Adam lr=0.0008, hidden 128/64, 19 epochs)
and features are imported verbatim.

The harness also loads DVMnet's shipped `vectors_{lnc,mi}.pkl` through a **restricted
unpickler** that permits only numpy array reconstruction and raises on anything else. These
are third-party pickles; static `pickletools` inspection found nothing but numpy, and both
load cleanly under the restriction.

## 3. Scripts → reported results

| Script | Produces | Needs |
|---|---|---|
| `reproduce_rnalight.py` | featurization check (max\|Δ\| < 1e-16) + AUROC 0.780 vs reported 0.783 | RNALIGHT_DIR |
| `exp_homology_split.py` | random CV 0.728; MMseqs2 GroupKFold 0.728 (id0.5) / 0.725 (id0.8); homolog coverage 1.6–5.2% | bundled `data/` |
| `exp_composition_confounds.py` | AUROC by feature set (dinucleotide-16 = 0.703 = 97% of the full model) | RNALIGHT_DIR |
| `exp_confound_matched.py` | length/GC-matched AUROC 0.689; 8-seed CV 0.730 ± 0.004 | RNALIGHT_DIR |
| `exp_study_effort.py` | strata AUROC 0.730/0.719/0.672/0.716 (no famous-few gradient) | RNALIGHT_DIR |
| `exp_dvmnet_topology.py` | degree-sum 0.750 under their protocol; 0.500 cold-start (model-free) | DVMNET_DIR |
| `exp_dvmnet_negatives.py` | degree-matched negatives 0.458 (model-free) | DVMNET_DIR |
| **`exp_dvmnet_model.py`** | **the ablation ladder: DVMnet's deep model, 5 conditions × 5 folds** → `results/dvmnet_model_results.json` | DVMNET_DIR |
| **`exp_dvmnet_negative_types.py`** | 55.5% of DVMnet's negatives are impossible pair types (analytic expectation 54.3%) | DVMNET_DIR |
| **`exp_dvmnet_stats.py`** | paired tests / CIs / Cohen's dz → `results/dvmnet_stats.json` | `results/` |
| `exp_lncptpred_audit.py` | benchmark probes: length match fails (0.528), GC leaks 0.628 | LncPTPred data |
| `exp_lncptpred_model.py` | model-level: 0.790 full, −0.031 under GC matching | LncPTPred data |
| `make_figures.py` | Figures 1–3 (`.png` + `.pdf`) | `results/` |
| **`factcheck_draft.py`** | asserts all 70 manuscript numbers against the results JSONs | `results/` |

## 4. The headline, reproduced

```bash
cd code
export DVMNET_DIR=/path/to/your/clone/DVMnet
python exp_dvmnet_model.py --out ../results/dvmnet_model_results.json   # ~70 min, CPU
python exp_dvmnet_stats.py
```

| condition | best epoch | final epoch | degree baseline (0 params) |
|---|---|---|---|
| verbatim (as released) | 0.8234 ± 0.0230 | 0.8021 ± 0.0364 | 0.7600 |
| random-uniform | 0.8028 ± 0.0168 | 0.7375 ± 0.0181 | 0.7502 |
| **random-degmat** | **0.5098 ± 0.0287** | 0.5065 ± 0.0283 | 0.4581 |
| cold-uniform | 0.6839 ± 0.0217 | 0.5803 ± 0.0870 | 0.4943 |
| cold-degmat | 0.5735 ± 0.0279 | 0.3924 ± 0.0217 | 0.3841 |

Fixing **only** the negative sampling — the published split left exactly as the authors
wrote it — takes the deep model from 0.803 to **0.510, i.e. chance**, in every one of five
folds (paired Δ = −0.2929, 95% CI [−0.3488, −0.2371], Cohen's dz = −6.51).

### What we do not claim

Neither control is clean, and they are biased in opposite directions: cold-uniform is
optimistic (popularity still leaks), and cold-degmat is pessimistic, because in a
positive-unlabelled setting degree-matched negatives are exactly the pairs most likely to
be *unobserved true interactions*. So **0.39–0.68 is a floor, not an estimate**, and the
released benchmark contains no information that can narrow it. The defensible claim is that
the reported 0.87 is not attainable under any leakage-controlled protocol we tested — not
that the model knows nothing. It beats the degree heuristic in every condition.

`results/dvmnet_model_results.json` is a **direct, single-run output** of
`exp_dvmnet_model.py` and carries per-fold arrays; `exp_dvmnet_stats.py` reads that JSON
rather than parsing the run log, so no published number depends on a rounded log value.

## 5. Changelog

- **v2.0.0** (2026-07-16) — DVMnet's **deep model** run under the full ablation ladder
  (v1.0.0 was model-free only); **LncPTPred** added as a third target; paired statistics;
  the negative-set validity probe (55.5% impossible pair types); `factcheck_draft.py`;
  per-fold arrays and cached embeddings for exact reproduction.
- **v1.0.0** (2026-07-02) — RNAlight + DVMnet, model-agnostic protocol diagnosis.

## 6. License

Audit code: MIT (see `LICENSE`). Upstream data retains its original license and is not
redistributed here.
