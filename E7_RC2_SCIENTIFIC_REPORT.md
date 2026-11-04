# E7-RC2 — Qualified Four-Family Annotation Panel: Results and Interpretation

**Status:** QUALIFIED collection, COMPLETE fixed-score and three-arm rescoring study, COMPLETE secondary E1 oracle ladder. 2026-11-04.

## Scientific purpose
This experiment assesses **annotation-panel sensitivity** of the existing fixed-audit repair and certification procedures. It uses the same AG News development (5,000) and holdout (2,600) items and benchmark labels as the earlier experiment, but changes the annotation panel to one source per model family: Qwen, Llama, Mistral and Phi. Original qualified Qwen and Llama model responses were retained. Mistral and Phi were fully recollected using the versioned constrained-output amendment E7-RC2 after their first-format collection failed the frozen 5% unusable-response gate.

**This is neither an independent dataset nor a clean causal intervention on model diversity.** Mistral/Phi use constrained choice decoding; Qwen/Llama retain the original direct/reason response contracts. The panel comparisons therefore conflate source composition and collection format. All conclusions are conditional on these fixed AG News items.

## Input/qualification audit

- Uploaded amended archive: `E7_RC2_qualified_results.zip`, 18 members; ZIP integrity passed.
- The RC2 manifest, frozen task hashes, and all 30,400 response hashes/IDs were independently checked using the original E7 collection for Qwen/Llama and the new RC2 collection for Mistral/Phi.
- Gold and text columns in the two joined CSV files exactly match the frozen original dataset files. Original Qwen/Llama responses are unchanged.
- Every model/panel combination meets the 5% invalid-response gate. Invalid totals: dev Qwen 21, Llama 11, Mistral 0, Phi 0; holdout Qwen 7, Llama 7, Mistral 0, Phi 0.
- All model revisions, Docker image digest, amended prompt contracts, and the pre-gold gate remain archived. The original failed Mistral/Phi collection is retained only as provenance, never used as the qualified source.

## Descriptive annotation quality

| Fixed AG News items | Old-panel DS accuracy | E7-RC2 DS accuracy | Old mean pairwise kappa | New mean pairwise kappa |
|---|---:|---:|---:|---:|
| Development, n=5,000 | 82.68% | 81.24% | 0.737 | 0.702 |
| Holdout, n=2,600 | 83.38% | 81.38% | 0.755 | 0.722 |

The four-family panel changes the error and disagreement structure; it does **not** improve baseline DS accuracy in these observed items. These are complete fixed-panel descriptive statistics, not estimates of performance across unseen datasets or annotator panels.

## Primary E7-RC2: new-panel factorial and practical rescoring

The prior frozen group-certificate rule, fixed G0/G1/G2 generators and S0/S1 scores, 400 total preprocessing audits (120 warm-up), 500 with-replacement PAC draws, 20 preprocessing seeds, and 10 paired PAC repetitions were retained. Arms share each generator's audits and PAC draws:

- **A**: same-audit DS labels and their baseline S0/S1 scores;
- **B**: certified repaired labels and **unchanged** baseline scores;
- **C**: certified repaired labels and class-conditional uncertainty **recomputed for the repaired predictions** using the same frozen DS posterior or warm-up text classifier;
- **O1**: gold-informed positive-net shortlisted group actions under unchanged baseline scores (retrospective only).

The frozen code/inputs and source hashes are in `results/E7_RC2_ANALYSIS_PROTOCOL.json`; analysis choices are an amendment made after historical panels/results were seen, not an original prospective preregistration.

**Completeness:** 120 generator–panel–seed preprocessing states; 60 accepted group actions; 10,000 PAC arm records; 2,400 paired A/B/C/O1 comparisons. All six numerical PAC spot checks, original evaluator vs optimized evaluator, and all paired accounting checks passed. Zero harmful accepted actions against each frozen reference were **observed**; the formal confidence argument is separate.

| Pool | Generator / score | Accepted states (of 20) | Label-only A−B | Joint A−C | Seed-bootstrap 95% CI for A−C |
|---|---|---:|---:|---:|---|
| Development | G0/S1 | 19 | −16.43 | **+29.71** | [+16.32, +43.82] |
| Development | G1/S1 | 15 | +2.89 | −14.62 | [−51.90, +17.39] |
| Holdout | G0/S1 | 12 | +2.79 | +3.76 | [−8.74, +15.51] |
| Holdout | G1/S1 | 3 | −1.04 | **−3.85** | [−9.66, 0.00] |

Both original-panel and new-panel S0 *incremental* joint repairs produce no material mean savings. All G2/S1 results are in `results/e7_rc2_seed_cluster_summary.csv`.

Positive numbers mean fewer purchased benchmark labels. G0/S1 development joint savings are statistically separated from zero by the **pointwise conditional seed bootstrap**, but amount to **0.594% of the 5,000-item pool**, below the previously declared practical target of 1%. The corresponding holdout G0/S1 interval includes zero. The main G1/S1 holdout response remains nonbeneficial on average, though small and uncertain (−3.85 labels compared with −19.97 in the original panel). No practical equivalence claim is made from zero-width bootstrap intervals.

**Operational cost warning:** the G0/S1 development same-audit pipeline buys 3,684.59 labels on average; joint repair brings that to 3,654.88, still above PAC-only at 3,430.51. Incremental repair benefit does not imply a competitive full pipeline. On the holdout, G0/S1 A costs 1,933.16 and C 1,929.40, compared with PAC-only 1,904.56.

### Certification/access changes with the panel

| Pool/generator | Original accepted states | New accepted states |
|---|---:|---:|
| Development G0 | 8 | **19** |
| Development G1 | 12 | **15** |
| Development G2 | 2 | **8** |
| Holdout G0 | 0 | **12** |
| Holdout G1 | 8 | **3** |
| Holdout G2 | 1 | **3** |

Thus the new panel changes the number and distribution of certifiable groups substantially, while the ability of accepted repairs to change costs remains dependent on scores and the certifier's threshold. Neither the 60 accepted actions nor the 2,400 PAC pairs constitute independent annotation panels. The raw panel contrast is descriptive because both panels use identical benchmark items, and only two fixed panel realizations exist.

## Secondary E1 oracle ladder on the new panel

The E1-style uncertainty warm-up is a different preprocessing allocation from the factorial; its oracles must not be interpreted as the same-arm effects. This additional sensitivity block was specified separately **after** the primary E7 analysis was underway and is exploratory. The O4 threshold-prefix intervention uses gold plus the selected baseline cutoff and therefore does not inherit the usual frozen-input PAC interpretation.

| Counterfactual (same-audit DS, scores fixed) | Development labels saved | Holdout labels saved |
|---|---:|---:|
| O1: positive-gain shortlisted groups | **0.00** | **0.00** |
| O2: every wrong prediction with vote disagreement | 105.61 | 99.06 |
| O3: every wrong prediction with four unanimous valid votes | 1,658.61 | 759.54 |
| O4: every error in the *observed* baseline test prefix | 777.37 | 440.79 |
| O5: every wrong prediction | **2,580.42** | **1,091.71** |

There were 40 preprocessing states and 3,200 arm records in this separate block. O1 had identical paired costs in all 400 tested calibration repetitions; O5 achieved the fixed-query floor in all repetitions. O5 savings equal 51.61% of the development pool and 41.99% of the holdout pool, **not** those percentages of baseline expert-label cost. These are optimistic, gold-informed interventions—not feasible cleaner performance bounds without their stated restrictions.

## Validity/interpretation

1. This analysis reproduces the historical computational *protocol* on an independently collected model-response panel, not on new items; it does not isolate source diversity from prompt/decoding changes.
2. The Qwen/Llama half of RC2 uses original qualified responses. Mistral/Phi use the versioned, constrained-choice RC2 responses. Original failed outputs were excluded.
3. Score and prediction vectors are frozen before independent PAC draws; reused stream identities make within-panel comparisons paired. The original and new panels have different audit trajectories by design.
4. The 20 preprocessing seeds quantify conditional algorithmic randomness on fixed labels. PAC repetitions are nested, not independent preprocessing units. Bootstrap intervals are pointwise, non-simultaneous, and can be degenerate.
5. Observed final error-target violations were rare but do not prove a statistical guarantee; the underlying mathematical PAC and group-CS arguments remain separate.
6. E1 O1 is accuracy-oriented over a restricted action set; it is not cost-optimal. O4 is an adaptive diagnostic. E5 rescoring is a distinct intervention that preserves same-audit comparisons but changes scores.
7. Do not merge these new repeated evaluations into an artificially enlarged dataset-independent inferential sample.

## Conclusion

> A separately collected four-family annotation panel on the same AG News items passed the response-quality gate after a versioned, gold-blind constrained-output recollection for Mistral and Phi. Under the unchanged group-repair protocol, the number of accepted actions increased in several configurations, but joint label repair and rescoring did not yield a reproducible practically important cost benefit on the frozen holdout. A gold-informed oracle continued to show substantial fixed-score savings potential. These observations demonstrate sensitivity to annotation-panel composition and response contracts, while preserving the distinction between candidate access, certification and cost conversion. They do not establish performance across other datasets or annotation systems.

## Reproducibility

See `E7_RC2_REPRODUCE.md`, `results/E7_RC2_ANALYSIS_PROTOCOL.json`, `results/E7_RC2_COMPLETION.json`, `results/E7_RC2_INDEPENDENT_RESULT_AUDIT.json`, `results/E7_RC2_E1_INDEPENDENT_AUDIT.json`, and `audit/independent_pre_gold_join_audit.json`. The E7 add-on requires the separately published compact v8.5 source for the corrected core implementation and original precomputed label-free text features.
