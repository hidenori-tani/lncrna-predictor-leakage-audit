Table 1. The four questions asked of each benchmark, the comparison that answers each, and the conclusion each comparison permits.

| question | comparison | conclusion permitted | conclusion not permitted |
|---|---|---|---|
| Q1. What does the benchmark reward that is not the biology the task names? | models restricted to node degree, node identity, length or composition, scored under the published protocol | a floor: on this benchmark, a score at or below it is not evidence of task-specific learning | that the published model uses that information |
| Q2. Does the published model's score survive when that information is neutralized? | the split fixed; negatives or the evaluation set changed so that the Q1 information no longer separates the classes (checked: the Q1 models built on that information alone fall to chance; models that combine it across the two members of a pair are reported as measured); every model family run through the same change | discrimination beyond that information, on a different and harder target | a corrected estimate of the published score; a size of inflation |
| Q3. Which generalization setting does the reported score describe? | the model and the negatives fixed; the split changed from edges to entities, or from random to homology-aware | whether the score describes prediction among seen entities or for unseen ones | inflation, unless the original paper claims the setting its protocol does not test |
| Q4. Does the protocol contain defects that bias the score for any target? | one defect removed at a time, everything else fixed | a demonstrated inflation, in AUROC units | — |

Table 2. The three audited predictors: task, model, data, and reported score.

| | RNAlight (Yuan et al., 2023) | DVMnet (Wei et al., 2025) | LncPTPred (Das et al., 2025) |
|---|---|---|---|
| task | nuclear/cytoplasmic localization | lncRNA–miRNA interaction | lncRNA–protein interaction |
| model | LightGBM on k-mer 3/4/5 | GAT + GCN, dual-view, multitask | LightGBM on 720 features |
| data | 3,412 train / 380 test lncRNAs | 284 lncRNA × 520 miRNA, 1,057 edges | 1,590,016 (protein, fragment) pairs |
| reported | AUROC 0.7833 | AUROC 0.87 (figure legend only) | ">0.9" (inequality only) |

Table 3. Model families run on each benchmark, and the information each can use.

| benchmark | model family | information it can use |
|---|---|---|
| DVMnet | degree sum (no parameters) | the two endpoints' degrees in the training graph |
| | popularity (logistic regression) | the two log-degrees and their product |
| | node identity (logistic regression) | which two molecules are involved; nothing about the pair |
| | composition (gradient boosting) | k-mer fractions (k = 1–3) and lengths of both sequences; no graph |
| | doc2vec (gradient boosting) | DVMnet's own node embeddings of both endpoints; no graph |
| | seed sites (no parameters) | canonical 7mer-m8 matches to the miRNA seed in the lncRNA: the pair |
| | seed-site family (logistic regression) | counts of five canonical site types and the lncRNA length: the pair |
| | DVMnet (published) | node embeddings and the training graph |
| RNAlight | length, GC, length + GC (logistic regression) | one or two bulk properties of the transcript |
| | dinucleotide (logistic regression) | sixteen dinucleotide fractions |
| | k-mer (logistic regression) | the 1,344 k-mer 3/4/5 frequencies, linearly |
| | k-mer nearest neighbors (25, cosine) | similarity to individual training transcripts |
| | RNAlight (published; gradient boosting) | the 1,344 k-mer frequencies |
| LncPTPred | GC fraction; length (single features) | one property of the fragment |
| | gradient boosting, earlier-version settings | the 720 released features, 200 trees |
| | gradient boosting, recorded settings | the 720 released features, the settings decoded from the released model |
| | LncPTPred (released trained model) | the 720 released features |

Table 4. Conditions run on each benchmark and the question each answers. Within a benchmark, conditions that answer different questions are never combined into one quantity.

| benchmark | condition | split | negatives or evaluation set | question |
|---|---|---|---|---|
| DVMnet | released | random edges | as released: uniform over all node pairs, impossible pair types included | Q1 |
| | uniform | random edges | uniform over valid lncRNA × miRNA non-edges | Q4 (impossible pairs removed) |
| | nested selection | as the condition | epoch chosen on a selection half, scored on a disjoint test half | Q4 (selection on the evaluation data removed) |
| | swap | random edges | the positives' own endpoints, re-paired | Q2 |
| | degree-weighted family (α = 0.5, 1; joint; hub-excluded) | random edges or lncRNA entities | endpoints drawn by degree | Q2, sensitivity |
| | cold-uniform | lncRNA entities | uniform over valid non-edges of the held-out lncRNAs | Q3 |
| | cold-swap | lncRNA entities | the held-out positives' own endpoints, re-paired | Q3 under Q2 |
| RNAlight | published | shipped train / test | — | Q1 |
| | cross-validation | random, five folds × five seeds | — | Q1 |
| | matched subset | random, five folds | transcripts matched on length and GC | Q2 |
| | homology-aware | MMseqs2 clusters at 50% and 80% identity | — | Q3 |
| | 200 resplits | random, shipped test size | — | Q4 context: no defect to remove; where the single published split falls |
| LncPTPred | single features | all rows | — | Q1 |
| | retrained, recorded settings | 80/20 stratified by protein and label | — | reproduction |
| | GC-matched evaluation | as above | validation rows matched on GC | Q2 |
| | feature blocks removed (previous version) | five folds, n = 400,000, earlier settings | — | supplementary: information used by the earlier-settings model |
| | length-matched evaluation | as above | validation rows matched on length within protein, as the paper declares; read against random rows with the same number per protein | Q4 (declared match restored) |
| | independent stopping | as above | stop on one validation half, score the other | Q4 (selection on the evaluation data removed) |
| | transcript-disjoint split | not expressible: no transcript or locus identifier | — | Q3 cannot be asked |
