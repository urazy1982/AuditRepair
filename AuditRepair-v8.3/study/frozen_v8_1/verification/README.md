# Independent implementation verification

This directory contains CPU checks of the statistical and engineering implementation. **These are implementation checks on specified synthetic fixtures, not E1–E4 study results and not proofs of the general guarantees.** No real study pool or result is loaded by either verification entry point.

## Accepted WSL reproduction

The uploaded WSL reports dated 4 November 2026 are preserved in [workstation_20261104/](workstation_20261104/), with their environment record, independent comparison and [acceptance note](workstation_20261104/ACCEPTANCE.txt). The main and larger-pool runs passed 305,547 and 60,005 assertions respectively, with zero failures. Their detailed results match the original reports exactly after normalizing package paths and excluding elapsed times and the Python build string. The five unique referenced source/script hashes match the current package. This is a repeated implementation check on the same frozen fixtures and seeds, not additional independent scientific evidence. No further routine verification run is needed for this stage.

## Run

From the `v8_experiments` directory:

```bash
python verification/test_verification.py
python verification/verify_factorial_gate.py
python verification/verify_guarantees.py --source-root . --profile full
python verification/verify_large_pool_pac.py --source-root .
```

The full entry point imports the vendored `audit_repair/core.py` and `audit_repair/pac.py` without editing them. Use `--source-root /absolute/path/to/source` for a different checkout. The separate AST inspection uses `run_certified_v7_9_reference.py`, an exact reference copy of the older runner; it does not pretend that this static inspection validates new score training. The new factorial orchestration has its own independent synthetic gate.

Pass `--source-root .` as shown when the extracted project directory has a different name from the original `v8_experiments` directory; the unchanged main verifier's default retains that original directory convention. The separate final acceptance audit reads completed study checkpoints, unlike the synthetic guarantee verifiers: see `numerical_acceptance_audit.json` and `numerical_acceptance_note.txt` for its distinct scope.

The main report is `verification_report.json`. It is checkpointed with `status: in_progress` while running. Only a final `passed_under_tested_conditions` or `failed` status is a completed result. The process exits nonzero on a deterministic assertion failure, an exception, or the specified simulation regression signal. Output records the exact source hashes, script hash, counts, parameter values, seed scheme, and limitations.

A smaller simulation run is available with `--profile quick`; the exhaustive small-population checks are retained. Individual repetition counts can be overridden with `--wor-reps`, `--search-reps`, and `--pac-reps`. The default full PAC setting is the frozen study's `epsilon=0.03`, `alpha_final=0.10`, `m=500`, and 10,000 repetitions for each of four frozen synthetic inputs, each with `N=240`. The first supplementary pass at `epsilon=0.05, m=160` and its matching script are archived under `history/` and must not be reported as validation at the tighter settings.

## Assumptions and independent checks

| Assumption / implementation concern | Check | Scope and limit |
|---|---|---|
| Predictable nonnegative betting stakes and correct without-replacement null direction | An independent scalar product calculation uses `fractions.Fraction` for null means, stakes, capital, running mean and variance. It is compared with scalar and vectorized source implementations. | Default rational reference has 41 points nested in the production 401-point grid. Set `--reference-grid 401` for a slower reference at every production point. |
| Time-uniform lower confidence bound and optional stopping on the remainder | Enumerate every distinct ordering over `{0, 0.5, 1}` for every `N=1,…,8` at alpha 0.05, 0.20 and 0.50; group sequences by population composition and divide integer event counts by exact multinomial counts. Track the maximum reported lower endpoint over every prefix. Reconstruct the remaining gain independently. | 164 populations, 492 population/alpha combinations, 9,840 complete orderings, 14,748 prefixes. The complete production 401-point grid is used. The rational reference is limited to its declared nested grid. |
| Grid inversion must not jump across an unrejected grid null | Exhaust all 256 masks on a nine-point grid with endpoint 1 unrejected; check the predecessor of the first unrejected null, sticky rejections after capital falls, and domination by every unrejected grid-null predecessor. | This checks the discrete prefix rule. It neither assumes monotonicity across nulls nor uses bisection on a nonmonotone rejection set. |
| Correct acceptance target | Independently reconstruct `N(2 lower−1)−(2 sum(X)−t)`; retain first strict-positive acceptance before exhaustion; assess its true unaudited remaining gain. | Simulations separately report numerical cases where a tiny positive floating-point margin corresponds to zero on the nominal rational grid. Reported floating-point lower endpoints are compared exactly to rational true means. |
| Candidate hypotheses are fixed before certification; prior information is handled correctly | Hand-specified dissent/tie/missing-label fixture; record freeze timing after warm-up; change only never-queried gold and compare audit trajectories; deliberately overlap candidates and require exclusion of prior group audits. | Runs actual group generation/search with a deterministic observable base estimator. It does not validate DS estimation quality. |
| The certified remainder is the action eventually applied | Exercise an accepted group; verify the final refit does not overwrite the remainder, audit gold overrides predictions, and unique changed-item accounting agrees. | Synthetic structural fixture plus actual search simulations. |
| Familywise repair target is `alpha_repair`, not the PAC target | Run 1,000 full searches for each of three fixed adversarial synthetic fixtures with zero/negative/positive-gain mixtures, warm-up, and overlap. Compute `P(any nonbeneficial accepted group)` using independently recomputed gains. | Conditional fixed synthetic populations; the base estimator is deterministic. This is not the E4 coupling sweep. |
| PAC first-failing threshold and exact unique-label cost | For 18 fixed input/draw fixtures, compute every threshold capital in scalar rational arithmetic and form the explicit union of owned, sampled and threshold-labeled indices. Inject one nonmonotone certificate mask to test threshold-selection control flow. | The hand oracle uses `epsilon=alpha=0.2` to exercise several branches with short paths; it is separately labeled from study settings. |
| Calibration uses frozen arrays and independent randomness | Read-only input arrays, before/after hashes, separate `SeedSequence` paths, and an AST check of the older runner's seed wiring. | Seed inspection checks wiring, not a universal independence theorem. It does not make arbitrary callers safe. |
| PAC final-error failure target is `alpha_final` | 10,000 production PAC calls for each of four frozen synthetic inputs: ordered errors, misranked errors, score ties, and repaired predictions with already owned gold. | Main report: `N=240`, 500 draws. Fixed-input repetitions do not measure annotation-panel variability. |
| New factorial chronology, pairing and costs | `verify_factorial_gate.py` instruments charged oracle access, warm-up model fitting, candidate freezing, final fill and full-gold diagnostics. It then runs the actual 19-arm orchestration for three PAC repetitions on a synthetic input while intercepting pool/cache reads. | Includes all three generators, 57 synthetic rows and independent queried-index cost unions; no real pool is read. |
| The verifier can detect a concrete defect | Unit test mutates PAC threshold cost by +1 in a temporary source copy and requires the full CLI to exit 1 with the correct deterministic failure IDs. | Temporary copies stay within this directory and are removed. The source under study is untouched. |

## Targeted larger-pool PAC check

`verify_large_pool_pac.py` reuses the independently checked PAC harness and writes `large_pool_pac_report.json`. It keeps the completed main script/report unchanged and does not repeat exhaustive enumeration. This targets a concrete limitation of the four main PAC fixtures: with `N=240` and `m=500`, calibration alone observes much of the pool.

The additional fixtures each have `N=5000`, 32 score levels, and exactly 170 fixed initial mistakes (3.4%). One distributes mistakes across scores using a fixed fixture seed; the other puts them at the lowest scores and includes 400 truthful already owned labels. Each receives 10,000 independent calibration repetitions with `m=500`, `epsilon=0.03`, and `alpha_final=0.10`. The report records actual retention and unique-label cost alongside failure counts and one-sided bounds. The inputs are fixed before calibration and do not come from any of the seven real study pools.

The completed larger-pool run took 162.614 seconds. Both specified fixtures had 60 violations in 10,000 repetitions, giving an individual one-sided 95% upper failure bound of 0.00743361 (0.7434%), below `alpha_final=0.10`. Mean automatically retained labels were 2,176.8692 of 5,000 for distributed errors and 15.3807 for misranked errors with 400 owned audits. The misranked condition was conservative in cost: only 89/10,000 repetitions saved any labels. These positive violation counts demonstrate that the check exercised the failure boundary; they are not failures of a guarantee that permits a 10% event probability.

`verification_overview.json` records the combined deliverable status and exact per-fixture counts. An earlier supplementary two-fixture boundary run, started before the requested 32-level/400-audit refinement, is archived under `history/`; it is not included in the primary reported counts. Its archived script can be run with this directory on `PYTHONPATH` and an explicit `--source-root`.

## The finite-grid statement

The draft's proposed comparison to an “exact bound found by bisection” is not valid for arbitrary nonmonotone rejection patterns. There may be an unrejected interval between two rejected grid points. For example, if all values below 1 are rejected except `[0.2, 0.3]`, then the grid `{0, 0.5, 1}` reports `0.5`, while the first continuously unrejected value is `0.2`. This is a counterexample to that proposed general comparison; it is not claimed to be a realizable betting path for this implementation.

The coverage argument available to the implementation is different: choose the first grid point at or above the true mean. If that single true null is never rejected, the prefix rule stays at or below its preceding grid point. This argument does not require monotone rejection across null values. The verifier checks that discrete implication and directly enumerates coverage in the stated finite setting. It makes **no claim that every continuous first-unrejected bound is within one grid step**.

## Probability statements and report statuses

The repair target is `alpha_repair=0.05`; the final error target is `alpha_final=0.10`. The formal union-bound budget for the conjunction of the two stage guarantees is `0.15`. The final-error guarantee alone still uses `alpha_final`; adding the repair budget to it would be the wrong target.

For each simulation fixture the report supplies one-sided 95% Clopper–Pearson lower and upper bounds. Its classification is:

| Condition | Status |
|---|---|
| Lower confidence bound exceeds the nominal failure budget | `fail` |
| Upper confidence bound is strictly below the nominal failure budget | `pass_with_margin` |
| Neither condition holds | `inconclusive` |

An upper bound above the target is not by itself a failure. Multiple fixture intervals are not a simultaneous confidence statement. A simulation flag can occur by chance and must be distinguished from a deterministic reproduced defect. Zero observed violations are reported as a count and a finite upper confidence bound, never as proof of zero risk.

## What remains outside these checks

The checks assume truthful already-owned gold, finite nonnegative scores on unowned items, admissible probability parameters, and frozen predictions/scores before calibration. They do not validate the utility of text features or repair proposals, broader distributions of annotation panels, or the new study's substantive hypotheses. They also do not replace a mathematical audit of the cited theorem and its assumptions. Numerical outcomes in this folder should appear in an implementation-verification appendix, not in the study's efficacy tables.
