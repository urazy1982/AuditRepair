#!/usr/bin/env bash
set -euo pipefail
ROOT="${E7_REPO_ROOT:-$PWD}"
OUT="${E7_OUTPUT_ROOT:-$ROOT/e7_collected}"
python3 "$ROOT/AuditRepair-v8.4/code/e7_validate_and_join.py" \
  --tasks "$ROOT/AuditRepair-v8.4/E7_tasks" \
  --responses "$OUT/responses" \
  --reference "$ROOT/AuditRepair-v8.3" \
  --out "$OUT/validated"
python3 - "$OUT" <<'PY'
import json,sys,hashlib,datetime
from pathlib import Path
out=Path(sys.argv[1]);report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
for p in sorted((out/'responses').glob('*.jsonl')):
 report['files'][p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
(out/'provenance').mkdir(exist_ok=True)
(out/'provenance'/'response_checksums.json').write_text(json.dumps(report,indent=2)+'\n')
PY
printf '\nValidated panel files under %s/validated\n' "$OUT"
