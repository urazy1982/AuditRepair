#!/usr/bin/env python3
"""Ensure every sealed release file is staged by Git (and nothing unsealed is staged)."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
try:
    lines=subprocess.check_output(['git','ls-files','-z'],cwd=root).split(b'\0')
except (subprocess.CalledProcessError,FileNotFoundError):
    raise SystemExit('git init and git add -A first; this script reads the index')
tracked={x.decode('utf-8','surrogateescape') for x in lines if x}
manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text(encoding='utf8'))
required=set(manifest['files'])|{'RELEASE_MANIFEST.json','RELEASE_SHA256SUMS'}
missing=sorted(required-tracked)
extra=sorted(tracked-required)
print('Staged files:',len(tracked),'expected:',len(required))
if missing: print('MISSING release files:',*missing[:40],sep='\n  ')
if extra: print('UNEXPECTED staged files:',*extra[:40],sep='\n  ')
if missing or extra: sys.exit(1)
print('GIT INDEX FILE COVERAGE: PASS (run verify_complete_release.py for worktree byte checks)')
