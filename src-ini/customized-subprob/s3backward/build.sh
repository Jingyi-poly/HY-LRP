#!/bin/bash
# Build espprc_cpp pybind11 extension (src-ini/customized-subprob/s3backward).
#
# Output: the legacy espprc_cpp extension plus an ABI-specific verified manifest.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" build_native.py "$@"
