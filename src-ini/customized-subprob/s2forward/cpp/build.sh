#!/bin/bash
# Build the subset_dp_kernel pybind11 extension (src-ini/customized-subprob/s2forward/cpp).
#
# Output: subset_dp_kernel.<EXT_SUFFIX>.so in this directory.  The Python DP
# (s2forward/subset_dp.py) uses it automatically when importable and
# falls back to its numpy implementation otherwise.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
PY_INCLUDES=$("$PYTHON_BIN" -m pybind11 --includes)
EXT_SUFFIX=$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT="subset_dp_kernel${EXT_SUFFIX}"
OUT_TMP="${OUT}.new"
SRC="subset_dp_kernel.cpp"

rm -f "$OUT_TMP"
trap 'rm -f "$OUT_TMP"' EXIT

echo "[build] compiling $SRC -> $OUT"
EXTRA_LDFLAGS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
    EXTRA_LDFLAGS=(-undefined dynamic_lookup)
fi

"${CXX:-g++}" -O3 -Wall -shared -std=c++17 -fPIC -fvisibility=hidden \
    -DNDEBUG -march=native \
    $PY_INCLUDES \
    "${EXTRA_LDFLAGS[@]}" \
    "$SRC" -o "$OUT_TMP"

mv -f "$OUT_TMP" "$OUT"
trap - EXIT

echo "[build] verifying import..."
"$PYTHON_BIN" -c "import sys; sys.path.insert(0, '.'); import subset_dp_kernel; print('  module:', subset_dp_kernel.__file__)"
echo "[build] OK"
