#!/usr/bin/env bash
set -euo pipefail
ROOT="${E7_REPO_ROOT:-$PWD}"
KIT="$ROOT/E7_v2_amendment_kit"
# validator writes pre-gold assessment first; it only accesses frozen gold after gate passes
python3 "$KIT/validate.py" --repo "$ROOT" --amendment "$ROOT/e7_rc2" --old "$ROOT/e7_collected" \
 --new "$ROOT/e7_collected_rc2" --out "$ROOT/e7_collected_rc2/validated"
python3 - "$ROOT" <<'PY'
from pathlib import Path
import zipfile,sys
root=Path(sys.argv[1]);dest=root/'E7_RC2_qualified_results.zip'
files=sorted(x for base in ('e7_rc2','e7_collected_rc2') for x in (root/base).rglob('*') if x.is_file())
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
 for x in files:z.write(x,x.relative_to(root))
print('Created',dest,'with',len(files),'files')
PY
