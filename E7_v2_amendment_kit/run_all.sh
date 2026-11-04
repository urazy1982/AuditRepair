#!/usr/bin/env bash
set -euo pipefail
ROOT="${E7_REPO_ROOT:-$PWD}"
KIT="$ROOT/E7_v2_amendment_kit"
mkdir -p "$ROOT/e7_rc2"
python3 "$KIT/prepare.py" --repo "$ROOT" --out "$ROOT/e7_rc2"
for FAM in mistral phi;do
  bash "$KIT/run_family.sh" "$FAM"
  docker rm -f "auditrepair-e7rc2-${FAM}"
done
bash "$KIT/validate_all.sh"
