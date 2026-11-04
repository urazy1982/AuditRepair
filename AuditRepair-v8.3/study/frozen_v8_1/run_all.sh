#!/usr/bin/env bash
# Reuse compatible checkpoints or run missing jobs, then validate full scope.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash run_all.sh [RESULTS_DIR [ANALYSIS_DIR]]' \
        'Defaults: results and analysis, relative to the package directory.' \
        'Environment: V8_PYTHON (explicit interpreter), V8_WORKERS (default 3).' \
        'Auto interpreter: active virtualenv, project .venv, python3, then python.' \
        'Existing compatible checkpoints are reused; new directories force fresh jobs.'
    exit 0
fi
if (( $# > 2 )); then
    printf '%s\n' 'Expected at most RESULTS_DIR and ANALYSIS_DIR; use --help.' >&2
    exit 2
fi

if [[ -n "${V8_PYTHON:-}" && "$V8_PYTHON" == */* && "$V8_PYTHON" != /* ]]; then
    # Resolve an explicit relative interpreter from the caller's directory.
    V8_PYTHON="$PWD/$V8_PYTHON"
fi
V8_SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
cd -- "$V8_SCRIPT_DIR"
V8_RESULTS="${1:-results}"
V8_ANALYSIS="${2:-analysis}"
if [[ -n "${V8_PYTHON:-}" ]]; then
    if ! command -v -- "$V8_PYTHON" >/dev/null 2>&1; then
        printf 'V8_PYTHON is not an executable interpreter: %s\n' "$V8_PYTHON" >&2
        exit 127
    fi
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    V8_PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x .venv/bin/python ]]; then
    V8_PYTHON="$PWD/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    V8_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    V8_PYTHON="$(command -v python)"
else
    printf '%s\n' \
        'Python was not found. Install Python 3.12 and create a virtual environment.' \
        'On Ubuntu 24.04: sudo apt install python3 python3-venv' \
        'Or set V8_PYTHON to an existing environment interpreter.' >&2
    exit 127
fi
V8_WORKERS="${V8_WORKERS:-3}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

printf 'Using Python interpreter: %s\n' "$V8_PYTHON"
"$V8_PYTHON" run_e1.py --workers "$V8_WORKERS" --out "$V8_RESULTS/E1"
"$V8_PYTHON" run_factorial.py --workers "$V8_WORKERS" --out "$V8_RESULTS/E2_E3"
"$V8_PYTHON" run_e4.py --workers "$V8_WORKERS" --out "$V8_RESULTS/E4"
"$V8_PYTHON" analyse_results.py --results "$V8_RESULTS" --out "$V8_ANALYSIS"
