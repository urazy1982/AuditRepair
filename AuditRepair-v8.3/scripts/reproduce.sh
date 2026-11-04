#!/usr/bin/env bash
# One WSL/Linux front door; never write into sealed main-study results.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="${1:-verify}"
if [[ "$MODE" == '-h' || "$MODE" == '--help' ]]; then
  cat <<'EOF'
Usage: bash scripts/reproduce.sh verify|verify-deep|main|amendment|all [NEW_OUTPUT_DIRECTORY]
All recompute modes put their outputs in a new, ignored directory, never in the frozen source.
Use Python 3.12 with requirements.txt. Override the interpreter with V83_PYTHON=/path/to/python.
EOF
  exit 0
fi
if (( $# > 2 )); then echo 'Too many arguments' >&2; exit 2; fi
if [[ -n "${V83_PYTHON:-}" ]]; then
    PYTHON="$V83_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    echo 'Python not found. Activate Python 3.12 venv or set V83_PYTHON.' >&2
    exit 127
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

audit_target() {
    local destination="$1"
    "$PYTHON" - "$ROOT" "$destination" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve()
if out==root or not out.is_relative_to(root/'reproductions'):
    raise SystemExit('For safety, outputs must be inside repo/reproductions/ (and never inside archived source)')
if out.exists():
    raise SystemExit(f'Choose a NEW output path, not an existing directory: {out}')
PY
}

if [[ "$MODE" == verify ]]; then
    echo '[v8.3] Verify top-level package and studies'
    "$PYTHON" "$ROOT/scripts/verify_package.py"
    "$PYTHON" "$ROOT/study/scripts/check_release.py"
    ( cd "$ROOT/study/frozen_v8_1" && "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -q )
    echo '[v8.3] Verification passed. No experiment rerun performed.'
    exit 0
fi
case "$MODE" in
  main|amendment|all|verify-deep) ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${2:-$ROOT/reproductions/${MODE}_${STAMP}}"
if [[ "$DEST" != /* ]]; then DEST="$PWD/$DEST"; fi
# Keep paths inside root/reproductions; prevents mutating sealed release.
audit_target "$DEST"
mkdir -p "$DEST"
echo "[v8.3] Mode $MODE; Python: $PYTHON; output: $DEST"
if [[ "$MODE" == main || "$MODE" == all ]]; then
    mkdir -p "$DEST/main"
    ( cd "$ROOT/study/frozen_v8_1" && V8_PYTHON="$PYTHON" bash run_all.sh "$DEST/main/results" "$DEST/main/analysis" )
    "$PYTHON" "$ROOT/scripts/compare_main.py" --actual "$DEST/main/analysis" --reference "$ROOT/study/frozen_v8_1/analysis" --output "$DEST/main/comparison.json"
fi
if [[ "$MODE" == amendment || "$MODE" == all ]]; then
    ( cd "$ROOT/study" && V82_PYTHON="$PYTHON" bash run_revision.sh --recompute "$DEST/amendment" )
fi
if [[ "$MODE" == verify-deep ]]; then
    mkdir -p "$DEST/verification"
    "$PYTHON" "$ROOT/study/frozen_v8_1/verification/verify_guarantees.py" --source-root "$ROOT/study/frozen_v8_1" --profile full --output "$DEST/verification/verification_report.json"
    "$PYTHON" "$ROOT/study/frozen_v8_1/verification/verify_large_pool_pac.py" --source-root "$ROOT/study/frozen_v8_1" --output "$DEST/verification/large_pool_pac_report.json"
fi
printf '[v8.3] Completed mode %s: %s\n' "$MODE" "$DEST"
