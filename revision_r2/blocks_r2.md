# Table legends of the paper, read by make_tables_r2.py (copied from the manuscript's blocks).

=== TABLE_LEGENDS ===
Table 1. The four questions asked of each benchmark, the comparison that answers each, and the conclusion each comparison permits.

Table 2. The three audited predictors: task, model, data, and reported score.

Table 3. Model families run on each benchmark, and the information each can use.

Table 4. Conditions run on each benchmark and the question each answers. Within a benchmark, conditions that answer different questions are never combined into one quantity.

Table 5. The DVMnet benchmark: every model family under each condition. AUROC, mean ± SD over five independent cross-validation partitions; for DVMnet the epoch is chosen on a selection half and the model scored on the disjoint test half. Learned baselines under every condition other than the released one are trained on negatives drawn from the training graph alone.

Table 6. DVMnet: paired changes in the nested AUROC between conditions, each assigned to the question it answers, and the selection bias (epoch chosen on the test half minus epoch chosen on an independent selection half). Changes and intervals are over the five partitions, paired by partition; only the Q4 rows measure inflation.

Table 7. The RNAlight benchmark: every model family under each protocol. AUROC; cross-validation values are the mean ± SD over five seeds of five-fold cross-validation.

Table 8. LncPTPred reproduced with the full dataset and the settings recorded in the authors' released model, and the comparisons run on the reproduction.

Table 9. What each question permits, benchmark by benchmark. The rows are not evidence about one another.

