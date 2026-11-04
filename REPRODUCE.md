# Reproduce AuditRepair — WSL/Linux and offline scientific results

Use a clean Ubuntu 24.04 / WSL installation with Python **3.12**. The annotation collection requires a CUDA GPU, but **all CPU analyses can use the bundled cached source labels and dataset records**.

## 0. Environment

```bash
cd /path/to/AuditRepair
sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv "$HOME/.venvs/auditrepair-release"
source "$HOME/.venvs/auditrepair-release/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r AuditRepair-v8.3/requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python --version
```

The pinned requirements are an **observed version closure, not a cryptographic wheel-hash lock**. See `docs/WORKSPACE_AND_ENVIRONMENTS.md` for source and WSL details.

## 1. Verify supplied data and results without regeneration

```bash
bash VERIFY_COMPLETE.sh --quick
bash VERIFY_COMPLETE.sh --full
```

The `--quick` check is suitable for CI. The `--full` check includes release-specific tests and may be CPU intensive. Both verify the supplied datasets/artifacts, not empirical generalization to new model panels. Full source checks use the immutable E7 response files; they never need Hugging Face credentials.

## 2. Recompute the original E1–E4 experiment (CPU)

```bash
cd AuditRepair-v8.3
bash scripts/reproduce.sh main reproductions/e1_to_e4_fresh
bash scripts/reproduce.sh amendment reproductions/v82_amendment_fresh
cd ..
```

The first command regenerates the main E1–E4 grid with 320 checkpoints / 42,200 evaluations and compares authoritative analysis CSVs. The second recomputes the corrected acceptance replay and operational-power/active-mechanism amendment. Their original filenames and frozen hashes are preserved. New output paths must not already exist.

## 3. Recompute E5, E6, E8 (CPU)

```bash
export V83_REFERENCE="$PWD/AuditRepair-v8.3"
cd AuditRepair-v8.4
mkdir -p rerun
python code/e5_rescoring.py --reference "$V83_REFERENCE" --out rerun/E5
python code/e5_event_inference.py --rows rerun/E5/e5_paired_rows.csv --out rerun/E5/e5_seed_event_uncertainty.csv
python code/e6_synthetic.py --reference "$V83_REFERENCE" --seed-stop 100 --out rerun/E6
python code/e6_seed_inference.py --results rerun/E6
python code/e8_numerical_audit.py
cd ..
```

E6 regenerates a 54-cell, 100-synthetic-seed-per-cell design with 16,200 policy states / 81,000 paired PAC records. For a cheaper **smoke test only**, use `--seed-stop 2 --out rerun/E6_smoke`. Do not treat smoke-test estimates as study findings.

## 4. Independently reproduce E7-RC2 from frozen collected responses (CPU)

```bash
bash VERIFY_E7_RC2.sh
bash RECOMPUTE_E7_RC2.sh e7_rc2_local_reproduction
```

The E7 add-on contains original 30,400 responses, original failed-gate evidence, the gold-blind Mistral/Phi RC2 amendment (15,200 responses), qualified joined panels, the frozen analysis protocol, and completed main/secondary checkpoints. The recomputation command refuses an existing output directory. It uses the **same 7,600 AG News examples as the original panel**, not new benchmark items. Main analysis: 120 preprocessing states, 10,000 PAC arm records and 2,400 paired evaluations. Secondary oracle: 3,200 rows.

## 5. Repeat annotation collection (optional GPU)

See `docs/E7_COLLECTION_AND_GOLD_ISOLATION.md` and the scripts `E7_launch_kit/` and `E7_v2_amendment_kit/`.

- Original 30,400-request attempt **failed its preset per-family format gate** for Mistral/Phi. Do not describe it as a qualified panel.
- Retain the qualified Qwen/Llama responses of the E7 collection; for RC2, collect Mistral/Phi with constrained `choice` decoding. Record the pinned Docker image digest and each model revision; the model-name readiness API is not revision evidence.
- Verify request hashes, HTTP response status and strict output parse **before opening benchmark gold labels**. Keep attempts separately versioned.

The release ships original responses and immutable source information, but does not ship model weights, Docker images or Hugging Face tokens. A new collection is a new stochastic run, not a byte-for-byte replay promise.

## 6. Historical material and integrity

`historical_support/AuditRepair_v7_9_code_data_results.zip` contains the earlier v7.9 code, data and results, kept **only to trace older supporting analyses**; do not run its old evaluator as the current corrected pipeline. See `docs/STATISTICAL_CONTRACTS.md` and `docs/PROVENANCE_AND_LIMITS.md` for the statistical contracts and residual limitations.

A release-wide SHA-256 manifest is at `RELEASE_SHA256SUMS` and is checked by `scripts/verify_complete_release.py`. Stored results are read-only; all scripts that recompute create separate output directories.
