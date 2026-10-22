"""Tables 5-9 and Supplementary Tables S1-S5 of R2, generated from the results (called by
make_numbers_r2.py). Tables 1-4 are design tables and are read from tables_static_r2.md."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fill_r2  # noqa: E402


def legends():
    """Table legends, from the TABLE_LEGENDS block that the manuscript prints too (one source)."""
    out = {}
    for para in fill_r2.blocks()["TABLE_LEGENDS"].split("\n\n"):
        para = " ".join(para.split())
        if para:
            out[int(para.split(".")[0].split()[1])] = para
    return out

FAM_ROWS = [("degree_sum", "degree sum (no parameters)"), ("popularity_lr", "popularity"),
            ("node_id_lr", "node identity"), ("composition_gbm", "composition"),
            ("doc2vec_gbm", "doc2vec"), ("seed_sites", "seed sites (no parameters)"),
            ("seed_family_lr", "seed-site family"), ("DVMnet", "DVMnet (nested)")]
DVM_COLS = [(("random", "verbatim"), "released (Q1)"), (("random", "uniform"), "valid pairs only (Q4)"),
            (("random", "swap"), "swap (Q2)"), (("cold", "uniform"), "unseen lncRNAs, uniform (Q3)"),
            (("cold", "swap"), "unseen lncRNAs, swap (Q3, Q2)")]


def ms(v, nd=3):
    return f"{v['mean']:.{nd}f} ± {v['sd']:.{nd}f}"


def pick(summary, name):
    for k in (name + "@trainonly", name):
        if k in summary:
            return summary[k]
    return None


def table5(base, r1all, r2):
    head = "| model | " + " | ".join(lab for _, lab in DVM_COLS) + " |"
    lines = [head, "|---|" + "---|" * len(DVM_COLS)]
    for key, lab in FAM_ROWS:
        cells = []
        for cond, _ in DVM_COLS:
            if key == "DVMnet":
                src = r1all.get(cond) if cond in (("random", "verbatim"), ("random", "uniform")) else r2.get(cond)
                cells.append(ms(src["summary"]["nested"]) if src else "—")
            else:
                s = base.get(cond)
                v = pick(s["summary"], key) if s else None
                cells.append(ms(v) if v else "—")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table6(F):
    rows = [("Q4", "impossible pair types removed: released → valid pairs only", "rel_uni"),
            ("Q2", "node-level information removed: valid pairs only → swap", "uni_swap"),
            ("Q3", "unseen lncRNAs: valid pairs only → unseen lncRNAs, uniform", "uni_cuni"),
            ("Q3 under Q2", "unseen lncRNAs: swap → unseen lncRNAs, swap", "swap_cswap")]
    lines = ["| question | comparison (DVMnet, nested AUROC) | paired change | 95% CI |", "|---|---|---|---|"]
    for q, lab, tag in rows:
        if f"dvm_d_{tag}" in F:
            lines.append(f"| {q} | {lab} | {F[f'dvm_d_{tag}']} | [{F[f'dvm_d_{tag}_lo']}, {F[f'dvm_d_{tag}_hi']}] |")
    for c, lab in (("rel", "released"), ("uni", "valid pairs only"), ("swap", "swap"),
                   ("cuni", "unseen lncRNAs, uniform"), ("cswap", "unseen lncRNAs, swap")):
        if f"dvm_sb_{c}" in F:
            lines.append(f"| Q4 | epoch chosen on the test half instead of a selection half: {lab} | "
                         f"{F[f'dvm_sb_{c}']} | [{F[f'dvm_sb_{c}_lo']}, {F[f'dvm_sb_{c}_hi']}] |")
    return "\n".join(lines)


RL_FAMS = [("length_lr", "length"), ("gc_lr", "GC"), ("length_gc_lr", "length + GC"),
           ("dinuc_lr", "dinucleotide (16)"), ("kmer_lr", "k-mer, logistic regression"),
           ("kmer_knn", "k-mer, 25-nearest-neighbor vote"), ("kmer_lgbm", "RNAlight (k-mer, gradient boosting)")]


def table7(rl):
    lines = ["| model | published split (Q1) | cross-validation, 5 seeds (Q1) | length- and GC-matched subset (Q2) "
             "| homology-aware, 50% (Q3) | homology-aware, 80% (Q3) |", "|---|---|---|---|---|---|"]
    for k, lab in RL_FAMS:
        lines.append(f"| {lab} | {rl['published'][k]:.3f} | {rl['cv'][k]['mean']:.3f} ± {rl['cv'][k]['sd']:.3f} | "
                     f"{rl['matched'][k]:.3f} | {rl['homology_05'][k]['mean']:.3f} | {rl['homology_08'][k]['mean']:.3f} |")
    rs = rl["resplit"]["kmer_lgbm"]
    lines.append("")
    lines.append(f"Single-split distribution (Q4), RNAlight's model over {rs['n']} random splits with the shipped test "
                 f"size: {rs['mean']:.3f} ± {rs['sd']:.3f} (5th–95th percentile {rs['p05']:.3f}–{rs['p95']:.3f}); "
                 f"the published split's {rs['published']:.3f} lies at the {rs['percentile_of_published']:.0f}th percentile.")
    return "\n".join(lines)


def table8(F):
    def g(k):
        return F.get(k, "—")
    lines = ["| model and evaluation | question | AUROC | log-loss | note |", "|---|---|---|---|---|",
             f"| released trained model, our features, all rows | reproduction | — | {g('lnc_rel_logloss_all')} | "
             f"{g('lnc_rel_logloss_expected')} expected from the model's own records |",
             f"| retrained, recorded settings, full data; stopped and scored on the validation split, as published | "
             f"reproduction | {g('lnc_rt_auc')} | {g('lnc_rt_logloss')} | authors: {g('lnc_valid_logloss')} at iteration "
             f"{g('lnc_best_iter')}; ours at {g('lnc_rt_best_iter')} |",
             f"| same fit, stopped on one validation half and scored on the other | Q4 | {g('lnc_rt_nested')} | — | "
             f"change {g('lnc_rt_sel_effect')} |",
             f"| same model, validation rows matched on length within protein (as declared) | Q4 | {g('lnc_rt_len_auc')} | — | "
             f"n = {g('lnc_rt_len_n')}; length alone {g('lnc_rt_len_only_matched')}; drawn again with "
             f"{g('lnc_lenctl_redraws')} other seeds {g('lnc_lenmatch_redraw_lo')}–{g('lnc_lenmatch_redraw_hi')} |",
             f"| same model, random validation rows with the matched rows' number of positives and of negatives of "
             f"every protein, drawn without regard to length | Q4 control | {g('lnc_lenctl_mean')} | — | "
             f"{g('lnc_lenctl_draws')} draws, 95% within {g('lnc_lenctl_lo')}–{g('lnc_lenctl_hi')}; length alone "
             f"{g('lnc_lenctl_len_only')} |",
             f"| same model, validation rows matched on GC | Q2 | {g('lnc_rt_gc_auc')} | — | "
             f"n = {g('lnc_rt_gc_n')}; GC alone {g('lnc_rt_gc_only_matched')} |",
             f"| earlier-version settings (200 trees), full data | gap | {g('lnc_r1full_auc')} | — | |",
             f"| recorded settings, n = 400,000 | gap | {g('lnc_o400_auc')} | {g('lnc_o400_logloss')} | |",
             f"| earlier-version settings, n = 400,000 (previous version) | gap | {g('lnc_r1_400k')} | — | five-fold CV |"]
    return "\n".join(lines)


def table_s2(rows):
    lines = ["| split | negatives | DVMnet, training negatives from the training graph | DVMnet, previous version | "
             "degree sum on the evaluation sets | node identity, previous training negatives | "
             "node identity, training graph only | selection bias, rerun | partitions |",
             "|---|---|---|---|---|---|---|---|---|"]
    lab = {"alpha0.5": "α = 0.5", "degmat": "α = 1", "joint": "joint", "nohub": "hub-excluded"}
    for r in rows:
        split = "published" if r["split"] == "random" else "unseen lncRNAs"
        cells = [split, lab[r["neg"]],
                 ms(r["new"]) if r["new"] else "—",
                 ms(r["old"], 4) if r["old"] else "—",
                 f"{r['deg']['mean']:.3f}",
                 f"{r['nid_global']['mean']:.3f}" if r["nid_global"] else "—",
                 f"{r['nid_train']['mean']:.3f}" if r["nid_train"] else "—",
                 f"{r['sb']:+.4f}".replace("-", "−") if r["sb"] is not None else "—",
                 str(r["repeats"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write(F, RAW, base, r1all, r2, rl, sens_rows):
    static = open(os.path.join(HERE, "tables_static_r2.md"), encoding="utf-8").read().rstrip()
    L = legends()
    for line in static.splitlines():          # Tables 1-4: the static file must carry the same legends
        if line.startswith("Table ") and line.split(".")[0].split()[1].isdigit():
            n = int(line.split(".")[0].split()[1])
            if " ".join(line.split()) != L.get(n):
                sys.exit(f"tables_static_r2.md: the legend of Table {n} differs from TABLE_LEGENDS in blocks_r2.md")
    parts = ["# Tables", "", "What lncRNA predictors measure — Hidenori Tani", "", static, ""]
    parts += [L[5], "", table5(base, r1all, r2), ""]
    parts += [L[6], "", table6(F), ""]
    if rl:
        parts += [L[7], "", table7(rl), ""]
    parts += [L[8], "", table8(F), ""]
    parts += [L[9], "", "<<TABLE9>>", ""]
    with open(os.path.join(HERE, "tables_r2.src.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    sup = ["# Supplementary Tables", "", "What lncRNA predictors measure — Hidenori Tani", "",
           "Table S1. Sampled DVMnet negatives by pair type under the released protocol, pooled over five folds "
           "(n = 1,057), against the analytic expectation.", "",
           "| pair type | measured | analytic expectation |", "|---|---|---|",
           "| lncRNA–lncRNA (impossible) | 13.4% | (284/804)² = 12.5% |",
           "| miRNA–miRNA (impossible) | 42.1% | (520/804)² = 41.8% |",
           "| impossible pair types | 55.5% | 54.3% |",
           "| valid lncRNA–miRNA | 44.5% | 45.7% |", "",
           "Table S2. The degree-weighted negative constructions of the previous version, rerun with training "
           "negatives drawn from the training graph alone (Q2, sensitivity). Evaluation sets are those of the "
           "previous version, drawn with degrees counted over all edges; the degree sum on them shows that they do "
           "not neutralize degree. The node-identity columns show the defect of the previous training negatives. The selection "
           "bias is that of the rerun, defined as in Table 6.", "",
           table_s2(sens_rows), "",
           "Table S3. LncPTPred single-feature baselines on all 1,590,016 rows (benchmark-level; no model).", "",
           "| declared matched? | single feature | AUROC | |", "|---|---|---|---|",
           "| yes (number) | class prior | positive rate 0.50000 | holds |",
           "| yes (strands) | strand only | 0.5000 ± 0.0009 | holds |",
           "| yes (length) | length only | 0.5281 ± 0.0012 | does not hold |",
           "| — | protein identity only | 0.4921 ± 0.0006 | holds |",
           "| no (undeclared) | GC only | 0.6284 ± 0.0007 | separates the classes |", "",
           "Table S4. LncPTPred feature blocks, earlier-version settings (200 trees), five-fold cross-validation at "
           "n = 400,000 (from the previous version).", "",
           "| feature set | features | AUROC |", "|---|---|---|",
           "| lncRNA side only (the released 351) | 351 | 0.7382 ± 0.0021 |",
           "| the released feature set (351 lncRNA + 369 protein) | 720 | 0.7926 ± 0.0016 |",
           "| 351 lncRNA + protein identity | 352 | 0.7978 ± 0.0017 |",
           "| released set minus the structure block and strand | 709 | 0.7898 ± 0.0017 |",
           "| released set minus strand | 719 | 0.7926 ± 0.0016 |", ""]
    with open(os.path.join(HERE, "supplementary_r2.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sup))
    print("wrote tables_r2.src.md and supplementary_r2.md")
