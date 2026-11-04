# Audit–Repair v8.1: completed experiments and reproducibility

This package contains the amended study protocol, cached inputs, CPU implementations, completed E1–E4 records, paired analyses, and independent implementation checks.

The experiments use cached annotations and texts. They need no model API, GPU, or network access once the Python environment and this package are available. The later extensions (E5–E8) are documented separately in `AuditRepair-v8.4/` and the E7 directories.

## Workstation reproduction accepted — 4 November 2026

The full E1–E4 rerun on WSL reproduced all 42,200 recorded cost/error and decision outcomes. All eight authoritative analysis CSVs are byte-identical to the reference results and to an independent recomputation. Some recorded thresholds and complete score-array hashes differ across environments; this is outcome reproduction, not universal bitwise identity.

The separate WSL verification upload also passed: **305,547 main-suite assertions and 60,005 larger-pool assertions, with zero failures**. Every detailed result, configuration, fixture hash, warning and limitation matches the reference verification reports after package-path normalization and exclusion of elapsed times and the Python build string. All five unique referenced source/script hashes match the current package files. WSL used Python 3.12.3, with the recorded NumPy, pandas, SciPy and scikit-learn versions matching the package pins.

The acceptance record, original WSL reports, environment record and independent comparisons are in [verification/workstation_20261104/](verification/workstation_20261104/). Start with [ACCEPTANCE.txt](verification/workstation_20261104/ACCEPTANCE.txt). The original reference reports remain at their existing paths. The known numerical zero-margin edge case remains documented; the separate audit found no affected acceptance among the 55 accepted study groups.

**The completed E1–E4 reproduction and WSL verification stage is closed.** No further routine rerun is required.

## Completed scope

All three execution blocks and the authoritative analysis are complete. See the [analysis manifest](analysis/analysis_manifest.json) and each block's `run_status.json`.

| Block | Completed design | Checkpoints | Evaluation records |
|---|---|---:|---:|
| [E1 oracle family](results/E1/run_status.json) | 7 pools × 20 preprocessing seeds × 10 paired PAC repetitions × 9 methods | 140 | 12,600 |
| [E2/E3 factorial](results/E2_E3/run_status.json) | 7 pools × 20 seeds × 10 repetitions × 19 methods | 140 | 26,600 |
| [E4 synthetic sweep](results/E4/run_status.json) | 4 settings × 10 seeds × 5 repetitions × 5 coupling weights × 3 methods | 40 | 3,000 |
| Total | Completed study evaluations | 320 | **42,200** |

These are **evaluation records, not 42,200 independent samples**. Methods share pools, audit states, and paired calibration draws. Repetitions reuse fixed annotation panels. The PAC-only arm intentionally appears in both real-data blocks and is checked for exact agreement.

The seven real pools contain **20,480 items across six datasets**. AG News contributes both a development pool and a holdout pool; these are not two independent datasets.

| Pool identifier | Items |
|---|---:|
| `ag_news_5000` | 5,000 |
| `ag_news_holdout` | 2,600 |
| `yelp_2600_star_alias` | 2,600 |
| `dbpedia` | 2,600 |
| `emotion` | 2,000 |
| `imdb` | 2,600 |
| `banking77` | 3,080 |

Dataset and source-archive provenance are recorded in [input_provenance.json](input_provenance.json). BANKING77 texts were joined to the cached annotation rows by item ID; both original and joined files are hashed in the protocol.

## Environment

The completed run used **Python 3.12.14** and the versions in [requirements.txt](requirements.txt): NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, and Matplotlib 3.10.8. Matplotlib is included for accompanying figures; the experiment runners do not need a display.

From this directory, using a Python 3.12.14 interpreter:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### WSL Ubuntu 24.04 setup

Ubuntu may provide `python3` without a `python` command. In WSL, from the extracted project directory, create an isolated environment and install the dependencies:

```bash
cd /mnt/c/audit_repair_v8_1
sudo apt update
sudo apt install -y python3 python3-venv
python3 -m venv "$HOME/.venvs/audit-repair-v8-1"
source "$HOME/.venvs/audit-repair-v8-1/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash run_all.sh
```

Use `python --version` to record the actual interpreter. Ubuntu's Python 3.12 patch version may differ from the recorded 3.12.14 environment; preserve the pinned dependencies and supplied feature cache. In a new WSL session, activate the environment again, or run `V8_PYTHON="$HOME/.venvs/audit-repair-v8-1/bin/python" bash run_all.sh`.

The launcher hotfix chooses an explicit `V8_PYTHON` first, then an active virtual environment, the project's `.venv`, `python3`, and finally `python`. The original downloaded launcher also accepts `V8_PYTHON=python3 bash run_all.sh` if that interpreter already has the required packages. An invalid explicit override fails clearly instead of silently choosing a different interpreter.

Installation may require a package index or a local wheel cache. Experiment execution itself is offline. These pins specify the principal Python dependencies, not a complete lock of transitive packages, BLAS libraries, or the operating system. Preserve the recorded environment and frozen feature arrays when assessing exact numerical reproduction.

## Validate and reproduce

All commands below run from the package directory. They do not collect new annotations.

### Validate the supplied records and recompute authoritative analysis

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python analyse_results.py
```

The analysis validates the protocol/input/core hashes, completed checkpoint identities and grids, costs, paired draws, the O0 identity, the O5 cost floor, and agreement of PAC-only results across the real-data blocks. It deliberately rejects incomplete full-scope results.

### Full experiment rerun or resume

```bash
python run_e1.py --workers 3
python run_factorial.py --workers 3
python run_e4.py --workers 3
python analyse_results.py
```

With the supplied completed checkpoints, these commands validate and reuse compatible records, then rebuild summaries. They do not automatically rerun completed preprocessing jobs. To execute all jobs afresh while preserving the supplied results, use new output directories:

```bash
python run_e1.py --workers 3 --out rerun/results/E1
python run_factorial.py --workers 3 --out rerun/results/E2_E3
python run_e4.py --workers 3 --out rerun/results/E4
python analyse_results.py --results rerun/results --out rerun/analysis
```

The optional [run_all.sh](run_all.sh) performs the same four steps and stops on the first error:

```bash
bash run_all.sh
bash run_all.sh rerun/results rerun/analysis
```

The wrapper defaults to three workers and limits common numerical-library thread counts to one when they are unset. Set `V8_PYTHON` to an interpreter path or `V8_WORKERS` to another process count if needed. Relative output paths are resolved from the package directory. An explicit relative interpreter path is resolved from the shell's directory before the wrapper changes into the package directory.

E1 and the factorial accept `--pools`, `--seed-start`, and `--seed-stop`; E4 accepts the seed options and always includes all four synthetic settings. The stop index is exclusive. Use a separate output directory for partial checks. Rerun the full requested scope before attempting the full-scope analysis.

### Resume and cache integrity

Each completed pool/seed job is written atomically. Repeating the same command reuses completed compatible checkpoints and computes missing ones. A protocol or implementation mismatch raises an error instead of mixing runs. The factorial additionally binds checkpoints to the feature-cache hash; E4 also checks its payload checksum. The final analysis verifies the exact full experimental grid, regardless of a runner's earlier resume checks.

`feature_cache/` contains the frozen TF-IDF/SVD features, neighbour arrays, and their metadata. Features remain aligned to original item IDs; the factorial remaps neighbours when it permutes pool rows. To rebuild features, use a separate working copy with its feature cache absent. Do not combine a regenerated cache with older factorial checkpoints: compare the array hashes and retain the new cache provenance.

Keep the supplied protocol and source files unchanged when reproducing this run. A changed method, parameter, input, or score construction belongs in a separately documented experiment, not an edited checkpoint or checksum.

## Frozen design and accounting

The controlling [protocol_v8_1.json](protocol_v8_1.json) has SHA-256:

```text
2931a624b1ce970ce15ce2a6d4205b3992aa42606fc4d972fb5a3db38b042f61
```

It is a prospective computational amendment made after review of historical v7.9 material, not a preregistration predating collection of the cached panels. `protocol_history/` retains the earlier local version; the top-level protocol and its checksum control these runs. Execution freeze files record the actual code and derived-input hashes before each block.

- **Budgets and targets:** at most 400 unique preprocessing labels; 500 uniform PAC draws with replacement; final error tolerance 0.03; final-error failure budget 0.10; repair-family failure budget 0.05. The 0.15 union-bound budget concerns the conjunction of the two stage guarantees, not final error alone.
- **Warm-up:** E1 and E4 retain the historical 120-label uncertainty-based warm-up. E2/E3 use the same 120-item uniform warm-up across all generators, included in the 400-label budget. `G0_uniform` is therefore a controlled variant, not an exact E1 replication.
- **Repair stage:** candidate memberships and alternatives freeze before group audits. The factorial excludes warm-up items from its disjoint populations. It uses at most ten groups, minimum size 30, at most 196 additional group-stage labels, and alpha 0.05 divided by the number shortlisted. The maximum per-group count is 80; the historical half-group rule stops at `t >= N/2`, including the ceiling for odd sizes. Acceptance is checked before futility and caps. Remaining labels go to uncertainty fill, excluding accepted populations.
- **Certificate target:** positive gain on the unaudited remainder relative to the frozen post-warm-up reference. This is distinct from gain relative to DS refitted on all audits, which is logged and compared separately.
- **Scores:** S0 is one minus the maximum same-audit DS posterior. S1 is one minus the warm-up-only text classifier's probability for the same-audit baseline label. S1 uses no source-vote features. Each cell reuses the identical score vector for no repair, repair, and its group-action oracle; it does not apply the historical `repair_conf` score update.
- **Sampling and cost:** the expert cost is the union of warm-up, group audits, uncertainty fill, distinct calibration draws, and threshold-reviewed items. Repeated draws and previously known labels are charged once. The real-data root seed is 20261030; separate streams control item permutations, preprocessing, and PAC draws. Arms share PAC streams within a repetition.
- **Threshold convention:** automatic acceptance uses `U < u_hat`; review uses `U >= u_hat`; the failing test includes `U <= u_hat`. The latter is the threshold-influencing region, including the reviewed boundary. Known audited items receive the internal score sentinel −1.

O1 is a privileged positive-net **group-action** oracle, not an unrestricted ceiling. O2 and O3 are nonnested corrections of disagreement and complete-unanimity errors; missing-vote cases can fall outside both. O4 depends on the baseline calibration threshold and is explicitly an adaptive retrospective diagnostic without the frozen-label PAC interpretation. O5 corrects every error for free and reaches the fixed audit-and-calibration query floor in this run. These oracle costs do not represent affordable learned repairs.

E4 always retains four source columns. Its four settings vary whether three or four sources share the injected confusion and whether the affected slice is 0.20 or 0.35 **within true class 0**, not overall class prevalence. Each fixed 5,000-item input uses five weights, `eta = 0, 0.25, 0.5, 0.75, 1`. Permuting scores within baseline-correctness strata preserves the exact conditional score multisets. Eta is a rank-mixing weight, not a claimed Pearson correlation. This gold-conditioned intervention is a mechanism experiment, not a deployable scoring method.

## Which results to use

**Use `analysis/` for real-pool effect estimates and confidence intervals.** It averages paired repetition differences within each of 20 preprocessing seeds and uses 4,000 bootstrap resamples. One resampling matrix is shared by every real-data contrast within a pool. Intervals concern algorithmic variability conditional on that fixed panel; they are pointwise, not simultaneous winner-selection intervals.

| Output | Purpose |
|---|---|
| `analysis/E1_effects.csv` | Oracle and practical baseline comparisons |
| `analysis/E2_E3_effects.csv` | Within-cell repair/oracle effects and comparisons with PAC alone |
| `analysis/E2_E3_primary_contrasts.csv` | Prespecified generator and score contrasts; secondary interaction |
| `analysis/E1_reach.csv`, `analysis/E2_E3_reach.csv` | Correction denominators, threshold reach, and available retention diagnostics |
| `analysis/*_observed_error.csv` | Descriptive final-error counts, not proofs of guarantees |
| `analysis/analysis_manifest.json` | Integrity checks, source checkpoint hashes, and bootstrap-matrix hashes |
| `results/E4/paired_differences.csv`, `results/E4/reach_intervals.csv` | **Authoritative E4 intervals**, already using paired seed resampling |
| `results/E4/coupling_diagnostics.csv` | Realized association, preserved score quality, and undefined-correlation reasons |

The real-data block-local `paired_differences.csv` and factorial `prespecified_contrasts.csv` retain their initial bootstrap Monte Carlo draws. Their real-pool intervals are superseded by `analysis/`; they are retained as execution artifacts. Raw checkpoint records remain the source evidence. Positive `mean_reviews_saved` in authoritative effect tables means saving; primary factorial contrasts use differences in incremental cost, where negative favors the first named intervention.

## Tests and independent verification

The completed validations are separate counts:

| Suite | Completed count | Location / command |
|---|---:|---|
| New E1, factorial, runner, and E4 unit/integration tests | 33 | `python -m unittest discover -s tests -p 'test_*.py' -v` |
| Original v7.9 tests | 12 | Separate original-source validation; see `verification/legacy_tests.log` |
| Verification-harness tests, including deliberate defect detection | 7 | `python verification/test_verification.py` |

The original test suite is not duplicated in `tests/`; rerunning it requires the original v7.9 source archive named in the provenance record. Some synthetic engineering fixtures use a controlled observable estimator in place of DS to isolate audit chronology and accounting. That substitution is explicit and is not a test of DS prediction quality.

See [verification/README.md](verification/README.md) and [verification_overview.json](verification/verification_overview.json) for detailed assumptions, reports, and limits. The main report includes exact small-population checks, 4,000 without-replacement simulations, 3,000 full searches, and 40,000 PAC repetitions on four fixed small inputs. A separate [larger-pool report](verification/large_pool_pac_report.json) adds two fixed 5,000-item inputs with 10,000 PAC repetitions each. All are implementation checks under specified conditions, excluded from the 42,200 study records; none proves a universal guarantee.

To rerun verification in a working copy, with portable source paths:

```bash
python verification/test_verification.py
python verification/verify_factorial_gate.py
python verification/verify_guarantees.py --source-root . --profile full
python verification/verify_large_pool_pac.py --source-root .
```

These commands replace their corresponding verification reports. Supplementary earlier settings remain in `verification/history/` and must not be described as validation of the primary settings.

The targeted [numerical acceptance audit](verification/numerical_acceptance_audit.json) checked all 55 accepted study groups: 11 in E1, 31 in E2/E3, and 13 in E4. None was accepted solely because a zero nominal-grid remainder margin became slightly positive in floating-point arithmetic. Minimum exact positive margins were 0.025, 0.01, and 0.05 respectively. This resolves the identified edge case for the completed acceptances; it does not prove general floating-point validity. The frozen core is retained, with a conservative arithmetic recommendation recorded separately in `verification/`.

## Package layout

| Path | Contents |
|---|---|
| `protocol_v8_1.json`, `.sha256`, `protocol_history/` | Controlling amendment and prior local freeze |
| `*_execution_freeze.json`, `*_execution.log` | Execution provenance and logs |
| `data/`, `input_provenance.json` | Cached source votes, benchmark reference labels, texts, and origin records |
| `audit_repair/` | Unmodified vendored statistical core and simulator |
| `factorial.py`, `experiment_common.py` | New preprocessing and shared validation/accounting |
| `run_e1.py`, `run_factorial.py`, `run_e4.py` | Experiment entry points |
| `analyse_results.py` | Full-scope validation and authoritative real-data analysis |
| `feature_cache/`, `results/`, `analysis/` | Frozen derived inputs, completed raw records, and analyses |
| `tests/`, `verification/` | New tests and independent verification evidence |

The study treats the cached benchmark reference labels as oracle truth. Its real-data conclusions are conditional on the observed panels, tested generators, score sources, and certifier. Broader generalization is outside the scope of this frozen package.
