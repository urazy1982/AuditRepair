#!/usr/bin/env bash
# Compatibility entry point for the completed v8.2 revision package.
set -euo pipefail
V82_WRAPPER_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
exec bash "$V82_WRAPPER_DIR/run_revision.sh" "$@"
