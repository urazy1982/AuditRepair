# Reproduce E7-RC2 from a clean v8.5 GitHub checkout

1. Obtain the GitHub release of AuditRepair v8.5 (or extract `AuditRepair_v8_5_GitHub_Compact.zip`) into one directory. Its root must contain `AuditRepair-v8.3/` and `AuditRepair-v8.4/`.
2. Extract the **contents** of `AuditRepair_E7_RC2_Results_Addon.zip` into that same root directory. The add-on contributes `e7_rc2/`, `e7_collected_rc2/`, `e7_collected/`, `results/`, `code/`, and this README. It does not replace or modify the frozen parent directories.
3. Use Python 3.12 and install the parent's pinned `AuditRepair-v8.3/requirements.txt` in a fresh virtual environment. Confirm versions; numerical score vectors can differ slightly in other Python/BLAS environments.
4. Run `bash VERIFY_E7_RC2.sh` to check release SHA-256 files, the original and RC2 response-to-task mapping, and the E7 results and PAC accounting.
5. To recompute results without overwriting original evidence, run `bash RECOMPUTE_E7_RC2.sh`. It writes to a new directory `e7_rc2_local_reproduction`; it **refuses to replace an existing directory**.
6. The collection itself requires the original model checkpoints and pinned Docker image; it is not rerun by the CPU reproducer. Source tasks and response files plus provider evidence are included for independent inspection.

The published main E7 protocol is in `results/E7_RC2_ANALYSIS_PROTOCOL.json` with the frozen digest in `results/E7_RC2_ANALYSIS_PROTOCOL.sha256`. The separate E1 supplementary oracle is in `results/E7_RC2_E1_SECONDARY_PROTOCOL.json`. Gold is accessed only after original/RC2 response-quality gates pass; the new-model collection itself is gold-blind.

Original v8.5 computational artifacts and E5/E6 results remain separate and unchanged. This is a supplemental E7 experiment, **not** a new original-preregistration claim nor a new dataset.
