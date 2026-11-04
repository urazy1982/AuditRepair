#!/usr/bin/env python3
"""Verify complete distribution SHA-256 identity and expected file set.

Does not attempt to prove the statistical guarantees or licenses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ROOT_FILES = {'RELEASE_MANIFEST.json', 'RELEASE_SHA256SUMS'}
EXCLUDED_DIRS = {'.git', '__pycache__', '.pytest_cache', '.venv', 'reproductions', 'rerun', 'local_outputs', 'local_runs'}


def known_files():
    out = {}
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(ROOT)
        if set(rel.parts) & EXCLUDED_DIRS:
            continue
        if rel.as_posix() in EXCLUDED_ROOT_FILES:
            continue
        if p.suffix == '.pyc':
            continue
        out[rel.as_posix()] = p
    return out


def main() -> int:
    doc = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text(encoding='utf8'))
    expected = doc['files']
    current = known_files()
    missing = sorted(set(expected) - set(current))
    unexpected = sorted(set(current) - set(expected))
    corrupt = []
    for rel, meta in expected.items():
        p = current.get(rel)
        if p is None:
            continue
        content = p.read_bytes()
        if len(content) != meta['size_bytes'] or hashlib.sha256(content).hexdigest() != meta['sha256']:
            corrupt.append(rel)
    large = [(rel, meta['size_bytes']) for rel, meta in expected.items() if meta['size_bytes'] >= 100_000_000]
    sums_path = ROOT / 'RELEASE_SHA256SUMS'
    observed_sums = {}
    for line in sums_path.read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        digest, sep, relpath = line.partition('  ')
        if not sep or relpath in observed_sums:
            raise SystemExit('Invalid or duplicate entry in RELEASE_SHA256SUMS')
        observed_sums[relpath] = digest
    expected_sums = {k: v['sha256'] for k,v in expected.items()}
    expected_sums['RELEASE_MANIFEST.json'] = hashlib.sha256((ROOT/'RELEASE_MANIFEST.json').read_bytes()).hexdigest()
    sha_list_error = sorted(k for k in set(observed_sums) | set(expected_sums)
                            if observed_sums.get(k) != expected_sums.get(k))
    counts = doc['research_accounting']
    errors = missing or unexpected or corrupt or large or sha_list_error
    report = {
        'status': 'passed' if not errors else 'failed',
        'files': len(expected),
        'release': doc['release'],
        'max_tracked_file_mb': round(max(x['size_bytes'] for x in expected.values())/1e6, 2),
        'expected_blocks': counts,
        'missing': missing[:30], 'unexpected': unexpected[:30], 'sha256_mismatch': corrupt[:30],
        'github_100mb_file_limit_violations': large[:10],
        'sha256sum_list_disagreements': sha_list_error[:30],
        'scope': 'Distribution identity only; independent scientific checks run separately.',
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
