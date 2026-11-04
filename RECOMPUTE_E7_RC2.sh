#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
out="${1:-e7_rc2_local_reproduction}"
test ! -e "$out" || { echo "REFUSING existing output: $out" >&2; exit 2; }
mkdir -p "$out"
python code/run_e7_rc2.py --root . --out "$out"
python code/run_e7_oracles.py --root . --out "$out"
python code/verify_e7_results.py --root . --out "$out"
PYTHONPATH=code python - <<PY
from pathlib import Path
import json
out=Path('$out')
assert json.loads((out/'E7_RC2_COMPLETION.json').read_text())['status']=='complete'
assert json.loads((out/'E7_RC2_E1_STATUS.json').read_text())['status']=='complete'
print('E7-RC2 CPU reproduction complete:',out)
PY
