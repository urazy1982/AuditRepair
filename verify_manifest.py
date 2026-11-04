#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
m=json.loads((root/'COMPACTION_MANIFEST.json').read_text())
errors=[]
for name,h in m['files'].items():
 p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:errors.append(name)
if errors:
 print('FAILED:',errors[:20]);sys.exit(1)
print('SHA-256 verified:',len(m['files']),'files; excluded redundant artifacts:',len(m['omitted_groups']))
