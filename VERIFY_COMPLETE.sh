#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"
MODE="${1:---quick}"
case "$MODE" in --quick|--full) ;; *) echo 'Usage: bash VERIFY_COMPLETE.sh [--quick|--full]' >&2; exit 2;; esac
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
else
    echo 'Missing Python; install Python 3.12 and create a venv.' >&2
    exit 127
fi
# Older independently sealed E7 scripts execute the program name "python".
# Provide a temporary alias in PATH when only python3 exists; do NOT edit sealed scripts.
ALIAS_DIR=""
if ! command -v python >/dev/null 2>&1; then
    ALIAS_DIR="$(mktemp -d)"
    ln -s "$PY" "$ALIAS_DIR/python"
    export PATH="$ALIAS_DIR:$PATH"
    trap 'rm -rf "$ALIAS_DIR"' EXIT
fi
export V85_PYTHON="$PY"
export V83_REFERENCE="$ROOT/AuditRepair-v8.3"
export PYTHONPATH="$V83_REFERENCE/study/corrected:$ROOT/AuditRepair-v8.4/code${PYTHONPATH:+:$PYTHONPATH}"
echo '[1/4] Release-level SHA-256 and file-set verification'
"$PY" scripts/verify_complete_release.py
"$PY" verify_manifest.py
echo '[2/4] Main original study input/checkpoint/reanalysis accounting'
"$PY" AuditRepair-v8.3/scripts/verify_package.py
echo '[3/4] New CPU experiments E5/E6/E8 data accounting'
"$PY" AuditRepair-v8.4/code/verify_release.py
echo '[4/4] Qualified E7-RC2 source / response / result accounting'
bash VERIFY_E7_RC2.sh
if [[ "$MODE" == '--full' ]]; then
    echo '[FULL] Original source, implementation and independent bounded numerical tests'
    (cd AuditRepair-v8.3 && "$PY" study/scripts/check_release.py)
    (cd AuditRepair-v8.3/study/frozen_v8_1 && "$PY" -m unittest discover -s tests -p 'test_*.py' -q)
    (cd AuditRepair-v8.4 && "$PY" tests/test_fast_pac.py && "$PY" tests/test_e6_design.py && "$PY" code/e8_numerical_audit.py)
fi
echo "COMPLETE RELEASE VERIFICATION PASSED ($MODE)"
