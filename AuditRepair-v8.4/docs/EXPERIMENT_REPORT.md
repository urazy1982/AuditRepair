# AuditRepair v8.4 — computational experiment report (4 November 2026)

**Status:** E5, E6 and E8 completed; E7 collection and out-of-panel validation NOT run. **Do not report E7 as evidence.** All inferences are conditional on the panels or synthetic mechanisms specified here. This release preserves the previous v8.3 evidence without modifying it.

## Scope and pre-specification

The accompanying frozen manifests record v8.4 E5 and E6 designs and the 166 parent input identities. E5 is a retrospective, previously analyzed annotation-panel extension; it was run under the separate recorded `frozen_e5_as_executed.json`. The complete E6 specification was hashed before its new synthetic outcomes. Neither panel-level independence nor pre-registered blind real-world confirmation is claimed.

## E5 — repairing and naturally recomputing confidence scores

A: DS labels and baseline uncertainty. B: certified repaired labels with unchanged uncertainty. C: certified repaired labels with the SAME previously trained score model evaluated at the new predicted label; scores frozen before PAC samples. All use identical paid audits and paired PAC draws. S0 evaluates the same-audit DS posterior at the new label; S1 evaluates the warm-up-trained text classifier at the new label.

Seven existing annotation panels × 3 generators × 2 scores × 20 existing preprocessing seeds × 10 PAC draws = **8,400 paired comparisons**. There are 62 active generator-score-seed states (31 distinct accepted repair states crossed with two scores). Data: `results/E5/e5_paired_rows.csv`, per-seed bootstrap: `e5_seedcluster_summary.csv`. All zero-acceptance states have exactly zero rescoring effect. The baseline/repair outcomes in active states were reexecuted and cross-checked with the parent archive.

| Panel / generator-score | Label-only mean saving A−B | Joint mean saving A−C | 95% whole-seed bootstrap interval for A−C |
|---|---:|---:|---:|
| AG News 5k, G0/S1 | -0.450 | **+8.685** | [+2.485, +16.520] |
| AG News 5k, G1/S1 | +2.015 | **-24.065** | [-59.276, +5.861] |
| AG News holdout, G1/S1 | +1.190 | **-19.970** | [-52.381, +2.106] |

Values are distinct expert labels saved; negative means additional purchases. The designated holdout-primary row has **200 nested PAC comparisons, 8/20 seeds with an accepted repair, 70 nonzero cost comparisons, 6 comparisons saving ≥26 labels, and 27 comparisons adding ≥26**. Mean final label error was 1.3098% for A and 1.3006% for C. Across all 8,400 records, observed final error >3% occurred 13 times for A and 15 times for C; these counts do not replace the risk guarantee.

The controlled alternative C performs **worse on average in the primary G1/S1 holdout cell**, but its interval includes zero. These historical seed/panel results are exploratory, and do not imply practical equivalence or a universal harmful effect of rescoring. For unaccepted states, A=B=C by design: E5 is mainly informative about the **31** previously accepted repairs.

## E6 — new gold-blind controlled synthetic candidate experiment

Synthetic populations contain N=5,000 items, observable group identifiers and observable frozen scores, and hidden synthetic truth used only when an audit buys a label or after the run for evaluation. Candidate ranking never reads gold. The 54-cell factorial varies group size {50,100,300}, planted mean gain {0,.2,.4}, score location {low,near,high}, and candidate shortlist {1,10}. All candidate pools have a 40% zero-gain share, with integer-exact realized gains. Each policy spends at most 196 group-stage audits within 400 total audits, with the original 80-audit/group and futility rules. Five PAC calibrations are nested within **100 newly generated independent synthetic populations per cell**. Compared policies are size ordering, a gold-blind fixed score-location proxy, and random ordering.

**Complete:** 54 cells × 100 seeds × 3 policies = **16,200 policy states**, with 5 nested PAC evaluations = **81,000 paired arm records**. All 5,400 cell-seed checkpoints were assembled from completed shard runs after matching every frozen protocol and cell identity. `results/E6_full` has authoritative combined raw records, state and summary tables and full-seed inference. All reported bootstrap intervals resample the 100 synthetic populations per cell; they do not treat 500 PAC draws as independent seeds.

Illustrative mean savings and probability of ≥1 accepted group **under the size-based policy, gain +0.4**:

| Group size / location / shortlist | Accepted seeds | Mean expert labels saved | Whole-seed bootstrap 95% interval |
|---|---:|---:|---:|
| 50 / low / 1 | 4/100 | 7.304 | [0.134,16.752] |
| 50 / low / 10 | 0/100 | 0.000 | [0,0] |
| 100 / low / 1 | 60/100 | 256.656 | [204.969,307.150] |
| 100 / low / 10 | 29/100 | 4.010 | [1.832,6.490] |
| 100 / high / 1 | 60/100 | **0.000** | [0,0] |
| 300 / low / 1 | 99/100 | 202.246 | [185.472,220.129] |
| 300 / high / 1 | 96/100 | **0.000** | [0,0] |

The high-score conditions demonstrate a constructive case of **successful certification without conversion to lower purchase cost**, while low-score conditions can yield large savings. Shared group audit budgets substantially reduce action probability/savings under a ten-group shortlist. An exactly zero observed cost is not evidence that rare effects are impossible; event-rate confidence bounds and all 162 policy-cell analyses are in `e6_seedcluster_inference.csv`.

The gold-blind score-location policy did not show a broad or robust improvement over size-based ranking. It is not described as a winning new algorithm. Its per-cell matched comparisons, including adverse results, are in `e6_policy_contrasts.csv`. In a single cell (size=100, gain=.4, low score, 10 groups) its mean incremental saving was +1.348 labels, bootstrap interval [0.246,2.630]. This isolated small advantage does not support an across-cell superiority claim.

**Limits:** These are purpose-built synthetic populations with observable planted groups, an idealized fixed base aggregator, and prespecified scores. Low/near/high refer to *predefined score bands*, not to threshold positions chosen using future PAC gold. E6 is not empirical proof that the deployed LLM candidate generators reach useful groups, nor independent evidence about future human annotators.

## E8 — independent numerical reference audit

- 585 checked small-domain without-replacement betting paths, including 310 pointwise null paths, with zero reported discrepancies; high-precision Decimal and exact Fraction references.
- 36 terminal PAC scenarios matched, including tied scores, degenerate scores, and repeated draws; zero observed mismatches.
- This is bounded mathematical/numerical evidence, NOT a proof of all finite-precision paths nor a verification against a downloaded upstream PAC Labels repository. Sticky group-grid and terminal PAC rejection remain distinct procedures.

## E7 — external-validity blocker

Eight **gold-blind** AG News worklists for Qwen, Llama, Mistral and Phi (development 5,000 and holdout 2,600) contain **30,400 new source-item requests**, with balanced prompt assignment. The collectors, pre-gold format gate and joining script are present in `E7_tasks` / `code`. No new model response had been produced at this stage, so the external-validity question **remained open** (it was later addressed by E7-RC2; see the repository root). Freeze and verify exact model revisions, collect all four families on the same items, reject any model/pool with >5% unusable outputs, then run a new paired evaluation before claiming a new-panel finding.

## Scientific implications

- **Experimental methodology:** E5 directly evaluates real deployment-like rescoring; E6 separates candidate power and conversion with positive controls and shared budgets.
- **Statistical rigor:** E6 has 100 independent *synthetic* seeds per cell and seed-level exact event intervals; E5 historical 20 seeds do not become new independent evidence. The analysis avoids pseudoreplication.
- **Originality:** a reusable cost-conversion diagnostics framework becomes more concrete; no new general certification theorem has been proved.
- **External validity:** no improvement could be claimed at this stage because E7 had not yet run.
