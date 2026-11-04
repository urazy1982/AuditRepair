#!/usr/bin/env python3
"""Create a deterministic complete GitHub ZIP from the frozen release manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,help='Final .zip destination, outside the repository')
a=parser.parse_args()
out=Path(a.output).resolve()
if root in out.parents or out==root:
    raise SystemExit('Output must be outside the release source directory')
if out.exists():
    raise SystemExit(f'Refusing to overwrite existing archive {out}')
info=json.loads((root/'RELEASE_MANIFEST.json').read_text())
files=sorted([*info['files'],'RELEASE_MANIFEST.json','RELEASE_SHA256SUMS'])
out.parent.mkdir(parents=True,exist_ok=True)
with ZipFile(out,'w',compression=ZIP_DEFLATED,compresslevel=8,allowZip64=True) as z:
    for rel in files:
        p=root/rel
        if not p.is_file():raise SystemExit('Missing file '+rel)
        zi=ZipInfo('AuditRepair/'+rel,date_time=(2026,11,4,12,0,0))
        zi.compress_type=ZIP_DEFLATED
        zi.external_attr=(0o100644<<16)
        z.writestr(zi,p.read_bytes(),compress_type=ZIP_DEFLATED,compresslevel=8)
with ZipFile(out) as z:
    bad=z.testzip()
    if bad: raise SystemExit(f'Corrupt member: {bad}')
    if len(z.infolist())!=len(files):raise SystemExit('ZIP entry count mismatch')
print('Wrote',out,'bytes',out.stat().st_size,'files',len(files))
print('sha256',hashlib.sha256(out.read_bytes()).hexdigest())
