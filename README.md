# What lncRNA predictors measure — audit code for three published lncRNA predictors

Reproduction code and derived results for:

> Tani H. *What lncRNA predictors measure: node degree, sequence composition, and the evaluation
> protocol.* Statistical Applications in Genetics and Molecular Biology (under review, manuscript
> SAGMB.2026.0060).

A reported benchmark score is a property of a model evaluated under a protocol — a split, a
negative-sampling scheme and a rule for choosing the checkpoint — not of the model alone. The paper
asks four questions of each of three benchmarks and answers each with one stated comparison:

| question | comparison | what it permits |
|---|---|---|
| Q1. What does the benchmark reward that is not the biology the task names? | models restricted to node degree, node identity, length or composition, scored under the published protocol | a floor for that benchmark and protocol |
| Q2. Does the published model's score survive when that information is neutralized? | the split fixed, the negatives or the evaluation set changed so that the Q1 models fall to chance; every model family run through the same change | performance on a different, harder target — not a corrected estimate |
| Q3. Which generalization setting does the reported score describe? | the model and negatives fixed, the split changed (edges → entities; random → homology-aware) | the scope of the reported score |
| Q4. Does the protocol contain defects that bias the score for any target? | one defect removed at a time | a demonstrated inflation |

On every benchmark several model families that differ in the information they can use run through the
identical comparisons on the same folds and the same sampled negatives, so that each benchmark is its
own control. Conclusions are specific to each benchmark; the three are not compared with one another.

Everything runs on a CPU.

## 1. Data — none of the audited data or models is redistributed

| predictor | task | upstream repository |
|---|---|---|
| RNAlight (Yuan et al. 2023) | nuclear vs cytoplasmic localization | https://github.com/YangLab/RNAlight |
| DVMnet (Wei et al. 2025) | lncRNA–miRNA interaction | https://github.com/liuliwei1980/DVMnet |
| LncPTPred (Das et al. 2025) | lncRNA–protein interaction | https://github.com/zglabDIB/lncptpred |

The harness reads a clone of each repository in `external/`, at the commit audited, and leaves it
unmodified; `external/README.md` gives the clone commands. This repository ships derived artifacts
only: the pooled RNAlight sequences and k-mer matrix, MMseqs2 clusters, PubMed co-occurrence counts,
and the **cached doc2vec node embeddings** used for every DVMnet run (gensim's doc2vec is not
deterministic; with the cache the DVMnet analyses reproduce exactly).

LncPTPred's released trained model (`models/models.zip`, two pickles) is read **without being loaded**:
`revision_r2/inspect_released_lncptpred_model.py` decodes it with an unpickler whose every class
reference resolves to an inert placeholder, so no code from the files can run. The decoded model is
third-party and is not included; the script regenerates it from your clone.

## Layout

The directories mirror the project in which the analyses were run, because each script reads its
inputs relative to its own directory:

| directory | contents |
|---|---|
| `code/`, `data/`, `results/` | v2 (the original submission): scripts, derived data and results |
| `repro/` | the analysis scripts of both revisions, the queues that ran them one at a time, the inputs they read next to themselves (embedding cache, MMseqs2 clusters), and the first revision's results with their logs |
| `revision/` | the first revision: every number and table it quotes, computed from the result files (`key_numbers.json`) |
| `revision_r2/` | this revision: results, provenance of the audited repositories and of the released LncPTPred model, and the scripts that compute every number, table and figure |
| `external/` | empty; clone the audited repositories here |

In the run logs, the location of the project on the author's computer has been replaced by `.`;
no other file has been edited.

## 2. Environment

```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```
Python 3.13.5; torch 2.13.0, torch_geometric 2.8.0, gensim 4.4.0, LightGBM 4.6, scikit-learn 1.9,
MMseqs2 18, ViennaRNA 2.7.2. On macOS, torch and LightGBM each load their own OpenMP runtime; the
baseline script sets `OMP_NUM_THREADS=1` to avoid the resulting deadlock.

DVMnet's CUDA-only code runs on CPU through two shims in `code/dvmnet_harness.py` that touch no model
math (`Tensor.cuda` becomes the identity; `cuda.set_device` a no-op). Its shipped `vectors_{lnc,mi}.pkl`
are loaded through a restricted unpickler that permits only numpy arrays.

## 3. Scripts and what they produce

| script | produces |
|---|---|
| `repro/exp_dvmnet_revision.py` | DVMnet under every condition: released, valid-pair uniform, **swap** (negatives that re-pair a set's own positives, so every molecule is as frequent among the negatives as among the positives), degree-weighted (α = 0.5, 1, joint, hub-excluded), each with the published split and with unseen lncRNAs; epoch chosen on a selection half, scored on the test half; five independent partitions. `--train-scope train` draws training negatives from the training graph only |
| `repro/exp_dvmnet_baselines.py` | the seven other model families on exactly the same folds and test negatives (checked fold by fold against the stored degree-sum values) |
| `revision_r2/verify_swap_draws.py` | replays every swap draw and checks it: no recorded interaction, no repeat, exact endpoint counts in every training set and test half, and re-drawn pairs equal to the maximum-flow minimum where exactness is impossible |
| `repro/exp_rnalight_baselines.py` | RNAlight families under the published split, repeated CV, a length- and GC-matched subset, homology-aware splits (MMseqs2, 50% and 80%), and 200 random splits with the shipped test size and class counts |
| `revision_r2/inspect_released_lncptpred_model.py` | the training settings, evaluation record and scaler of the released LncPTPred model (`revision_r2/provenance/released_lncptpred_model_report.json`) |
| `repro/exp_lncptpred_faithful.py` | LncPTPred: features for all 1,590,016 rows from the released feature code; the released model scored on them; retraining with the recorded settings by the recorded 80/20 procedure; stopping on one validation half and scoring the other; GC- and length-matched evaluation |
| `repro/count_released_heldout_negatives.py` | how often DVMnet's released training sampler draws a held-out interaction as a negative: the released call replayed over the five partitions, five folds and nineteen epochs, without training (`revision_r2/results/released_heldout_negatives.json`) |
| `repro/lncptpred_length_match_control.py` | the control for the length-matched LncPTPred evaluation: random subsets with the matched subset's number of positives and of negatives of every protein, drawn without regard to length (200 draws), and the matched subset redrawn with 20 other seeds (`revision_r2/results/lncptpred_length_match_control.json`) |
| `repro/run_queue_r2.sh` | runs the above one at a time at the lowest priority (`repro/run_queue.sh` for the first revision) |
| `repro/exp_lncptpred_exact.py` | the first revision's LncPTPred features rebuilt from the released code, on subsamples |
| `revision/make_tables.py` | every number and table of the first revision (`revision/key_numbers.json`), from the result files only |
| `revision_r2/make_numbers_r2.py`, `revision_r2/make_tables_r2.py` | every number and table of the paper (`revision_r2/key_numbers_r2.json`), from the result files only |
| `revision_r2/make_fig_*.py` | Figures 1–3 (`revision_r2/figures/`) |

Earlier analyses that the paper still quotes (RNAlight reproduction, homology coverage, study effort,
LncPTPred single features and feature blocks) are produced by the scripts of v2, kept in `code/`.

## 4. Changelog

- **v3.0.0** (2026-10) — second revision. Four explicit questions; several model families on every
  benchmark; swap negatives; DVMnet's degree-weighted conditions rerun with training negatives drawn
  from the training graph only (the earlier runs drew them with degrees counted over all interactions,
  held-out ones included, and under the entity-disjoint split over held-out lncRNAs as well);
  LncPTPred reproduced with the full dataset and the settings recorded in the released model. Also
  deposits the scripts and results of the first revision (2026-09), which no earlier version
  contained. The readings of v2 that this release withdraws are listed in the paper's response to
  the reviewers.
- **v2.0.1** (2026-07-16) — figure layout fixes.
- **v2.0.0** (2026-07-16) — DVMnet's deep model under protocol changes; LncPTPred added.
- **v1.0.0** (2026-07-02) — RNAlight and DVMnet, model-free.

## 5. License

Audit code: MIT (see `LICENSE`). Upstream data and models keep their original terms and are not
redistributed.
