"""Record the complete content identity of a reviewed revision snapshot.

Maintainer packaging step, not part of scientific recomputation. Run only
after the reports and evidence have been reviewed. This cannot authenticate
an external release; it supplies exact file identities for integrity checks.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
IGNORED_PARTS={"__pycache__", ".pytest_cache"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(path):
    relative=path.relative_to(ROOT)
    return path.is_file() and not (set(relative.parts)&IGNORED_PARTS) and not relative.parts[0].startswith('reproduction_')


def main():
    source={}
    for path in sorted(ROOT.rglob('*')):
        if not eligible(path):
            continue
        rel=path.relative_to(ROOT)
        if rel.parts[0]=='frozen_v8_1':
            continue
        if (path.suffix in {'.py','.sh'} or str(rel) in {'protocol_v8_2.json','requirements-experiments.lock'}):
            source[str(rel)]=sha(path)
    revision=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    code={"identity_type":"sha256 of canonical sorted source-path to file-hash mapping",
          "revision_sha256":revision,"git_commit":None,
          "note":"No external Git commit is invented. Frozen v8.1 identities are recorded separately.",
          "files":source}
    (ROOT/'provenance/code_revision.json').write_text(json.dumps(code,indent=2)+'\n')
    excluded={'RELEASE_MANIFEST_SHA256.json','results/release_check.json'}
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if eligible(p) and str(p.relative_to(ROOT)) not in excluded}
    result={"schema_version":1,"release":"Audit Repair v8.2 bounded experimental revision",
            "sealed_utc":datetime.now(timezone.utc).isoformat(),
            "protocol_sha256":sha(ROOT/'protocol_v8_2.json'),"code_revision_sha256":revision,
            "excluded_from_hashes":sorted(excluded),"files":files}
    (ROOT/'RELEASE_MANIFEST_SHA256.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"files":len(files),"code_revision_sha256":revision,
                      "release_manifest_sha256":sha(ROOT/'RELEASE_MANIFEST_SHA256.json')}))


if __name__=='__main__':
    main()
