#!/usr/bin/env bash
# The default verifies the completed revision; --recompute creates a separate run.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash run_revision.sh [--recompute [OUTPUT_DIR]]' \
        'Default: verify supplied hashes, tables, and nine focused regression tests.' \
        '--recompute: repeat the targeted acceptance replay, active analysis, and power study in a NEW directory.' \
        'Environment: V82_PYTHON (explicit interpreter), V82_WORKERS (default 2).' \
        'Interpreter fallback: active venv, project .venv, python3, python.'
    exit 0
fi
if (( $# > 2 )) || { (( $# > 0 )) && [[ "$1" != "--recompute" ]]; }; then
    printf '%s\n' 'Use --help for supported arguments.' >&2
    exit 2
fi
V82_CALLER_DIR="$PWD"
if [[ -n "${V82_PYTHON:-}" && "$V82_PYTHON" == */* && "$V82_PYTHON" != /* ]]; then
    V82_PYTHON="$V82_CALLER_DIR/$V82_PYTHON"
fi
if [[ -n "${2:-}" ]]; then
    V82_OUTPUT="$2"
    if [[ "$V82_OUTPUT" != /* ]]; then V82_OUTPUT="$V82_CALLER_DIR/$V82_OUTPUT"; fi
fi
V82_SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
cd -- "$V82_SCRIPT_DIR"
if [[ -n "${V82_PYTHON:-}" ]]; then
    if ! command -v -- "$V82_PYTHON" >/dev/null 2>&1; then
        printf 'V82_PYTHON is not an executable interpreter: %s\n' "$V82_PYTHON" >&2
        exit 127
    fi
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    V82_PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x .venv/bin/python ]]; then
    V82_PYTHON="$PWD/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    V82_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    V82_PYTHON="$(command -v python)"
else
    printf '%s\n' 'Python was not found. Use a Python 3.12 virtual environment or set V82_PYTHON.' >&2
    exit 127
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
V82_WORKERS="${V82_WORKERS:-2}"
printf 'Using Python interpreter: %s\n' "$V82_PYTHON"
"$V82_PYTHON" scripts/check_release.py
if [[ "${1:-}" == "--recompute" ]]; then
    V82_OUTPUT="${V82_OUTPUT:-$PWD/reproduction_$(date -u +%Y%m%dT%H%M%SZ)}"
    # Resolve existing symlink parents before protecting all released trees.
    "$V82_PYTHON" - "$PWD" "$V82_OUTPUT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
destination=Path(sys.argv[2]).resolve()
protected=('frozen_v8_1','corrected','results','docs','tests','scripts','references','provenance')
for name in protected:
    if destination.is_relative_to(root/name):
        raise SystemExit(f'Reproduction destination lies inside protected release directory: {name}')
if destination.exists():
    raise SystemExit(f'Choose a new output directory; this path already exists: {destination}')
PY
    if [[ -e "$V82_OUTPUT" ]]; then
        printf 'Choose a new output directory; this path already exists: %s\n' "$V82_OUTPUT" >&2
        exit 2
    fi
    mkdir -p -- "$V82_OUTPUT"
    "$V82_PYTHON" scripts/capture_environment.py --destination "$V82_OUTPUT"
    "$V82_PYTHON" scripts/replay_corrected_acceptance.py --workers "$V82_WORKERS" --out-dir "$V82_OUTPUT"
    "$V82_PYTHON" scripts/analyze_active_repairs.py --out "$V82_OUTPUT"
    "$V82_PYTHON" scripts/run_operational_power.py --workers "$V82_WORKERS" --out "$V82_OUTPUT"
    printf 'Reproduction complete: %s\n' "$V82_OUTPUT"
fi
