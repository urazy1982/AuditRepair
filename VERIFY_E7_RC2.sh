#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for d in AuditRepair-v8.3 AuditRepair-v8.4 e7_rc2 e7_collected_rc2 e7_collected results; do
  test -d "$d" || { echo "MISSING $d: extract E7 add-on into a v8.5 repo root" >&2; exit 2; }
done
sha256sum -c E7_RC2_SHA256SUMS
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python code/verify_rc2_independent.py
python code/verify_e7_results.py --root . --out results
python - <<'PY'
from pathlib import Path
import pandas as pd,json
r=Path('results')
a=pd.read_csv(r/'e7_rc2_e1_oracle_rows.csv')
assert len(a)==3200 and a.groupby(['pool','seed','rep']).size().eq(8).all()
w=a.pivot(index=['pool','seed','rep'],columns='method',values='cost')
assert (w['O1_shortlist']==w['same_audit_ds']).all()
assert (a.loc[a.method=='O5_all_errors','threshold_new']==0).all()
assert (a.loc[a.method=='O5_all_errors','error']==0).all()
assert (a.cost==a.pre_audits+a.calibration_new+a.threshold_new).all()
print('E7 RC2 secondary oracle ladder: PASSED')
PY
printf 'E7-RC2 verification COMPLETE\n'
