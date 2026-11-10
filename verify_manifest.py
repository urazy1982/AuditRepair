#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
m=json.loads((root/'COMPACTION_MANIFEST.json').read_text())
errors=[]
documentation_drift=[]
for name,h in m['files'].items():
 p=root/name
 if not p.is_file():
  errors.append(name)
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=h:
  if name == 'README.md':
   documentation_drift.append(name)  # Editable root documentation; original hash stays in manifest.
  else:
   errors.append(name)
if errors:
 print('FAILED:',errors[:20]);sys.exit(1)
print('SHA-256 verified:',len(m['files'])-len(documentation_drift),'unchanged files; editable README drift:',documentation_drift,'; excluded redundant artifacts:',len(m['omitted_groups']))
