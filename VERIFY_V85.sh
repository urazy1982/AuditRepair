#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${V85_PYTHON:-$(command -v python3 || command -v python)}"
"$PY" verify_manifest.py
export V83_REFERENCE="$ROOT/AuditRepair-v8.3"
export PYTHONPATH="$V83_REFERENCE/study/corrected:$ROOT/AuditRepair-v8.4/code${PYTHONPATH:+:$PYTHONPATH}"
( cd AuditRepair-v8.3 && "$PY" scripts/verify_package.py && "$PY" study/scripts/check_release.py )
( cd AuditRepair-v8.4 && "$PY" code/verify_release.py && "$PY" tests/test_fast_pac.py && "$PY" tests/test_e6_design.py && "$PY" code/e8_numerical_audit.py )
echo 'v8.5 compact release verification passed.'
