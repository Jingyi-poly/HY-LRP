#!/bin/bash
# Build stage2_bp_cpp pybind11 extension.
#
# Output: stage2_bp_cpp.<EXT_SUFFIX>.so in current directory.
#
# Requires:
#   - pybind11 (pip install pybind11)
#   - Gurobi C++ API. If GUROBI_HOME is not set, this script tries
#     `module load gurobi` and then falls back to common /home/gurobi paths.
#   - g++ with C++17 support
set -euo pipefail

cd "$(dirname "$0")"
OS_NAME="$(uname -s)"

if [[ -z "${GUROBI_HOME:-}" ]]; then
    # Some clusters expose Gurobi through environment modules. In non-login
    # shells the `module` function may need to be sourced first.
    if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh || true
    fi
    if command -v module >/dev/null 2>&1; then
        module load gurobi >/dev/null 2>&1 || true
    fi
fi

if [[ -z "${GUROBI_HOME:-}" ]]; then
    # Last-resort discovery for the local cluster layout. Prefer newer versions
    # by sorting reverse lexicographically.
    for d in $(ls -d /home/gurobi/*/linux64 2>/dev/null | sort -Vr); do
        if [[ -d "$d/include" && -d "$d/lib" ]]; then
            export GUROBI_HOME="$d"
            break
        fi
    done
fi

if [[ -z "${GUROBI_HOME:-}" && "$OS_NAME" == "Darwin" ]]; then
    # Standard Gurobi macOS installer layout.  Reverse lexical order selects
    # the newest normal versioned installation name.
    for d in $(ls -d /Library/gurobi*/macos_universal2 2>/dev/null | sort -r); do
        if [[ -d "$d/include" && -d "$d/lib" ]]; then
            export GUROBI_HOME="$d"
            break
        fi
    done
fi

if [[ -z "${GUROBI_HOME:-}" ]]; then
    echo "[build] ERROR: GUROBI_HOME not set and Gurobi could not be discovered."
    echo "        Try: module load gurobi"
    exit 1
fi

# Detect Gurobi version (-lgurobiXYZ where XYZ is e.g. 130).
GRB_LIB=""
if [[ "$OS_NAME" == "Darwin" ]]; then
    GRB_GLOB=("$GUROBI_HOME"/lib/libgurobi[0-9]*.dylib)
else
    GRB_GLOB=("$GUROBI_HOME"/lib/libgurobi[0-9]*.so)
fi
for f in "${GRB_GLOB[@]}"; do
    [[ -e "$f" ]] || continue
    bname="$(basename "$f")"
    bname="${bname#lib}"
    bname="${bname%.so}"
    bname="${bname%.dylib}"
    # Skip variants like gurobi130_light — we want the main library.
    [[ "$bname" == *_* ]] && continue
    GRB_LIB="$bname"
    break
done
if [[ -z "$GRB_LIB" ]]; then
    echo "[build] ERROR: no versioned libgurobi shared library found under $GUROBI_HOME/lib"
    exit 1
fi
echo "[build] using Gurobi: -l$GRB_LIB  (GUROBI_HOME=$GUROBI_HOME)"

PYTHON_BIN="${PYTHON_BIN:-python}"
PY_INCLUDES=$("$PYTHON_BIN" -m pybind11 --includes)
EXT_SUFFIX=$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT="stage2_bp_cpp${EXT_SUFFIX}"
OUT_TMP="${OUT}.new"

# Backup existing .so only when explicitly requested. Automatic builds should
# not litter src/price with timestamped backups.
if [[ "${BP_BUILD_BACKUP:-0}" == "1" && -f "$OUT" ]]; then
    cp "$OUT" "${OUT}.bak.$(date +%H%M%S)"
fi

# Compile atomically: a compiler/linker failure must not destroy the last known
# working extension.  Stale ABI variants are removed only after the new module
# has linked successfully.
rm -f "$OUT_TMP"
trap 'rm -f "$OUT_TMP"' EXIT

# Detect pip-installed Gurobi (provides libgurobi130.so that accepts PIP
# licenses when paired with grb_pip_shim.so).  If found, link the dynamic
# libgurobi*.so from the pip path so the pybind module can run under the PIP
# license shim; the static C++ wrapper still comes from GUROBI_HOME.
GRB_PIP_LIB=""
PIP_GRB_DIR=$("$PYTHON_BIN" -c "
try:
    import gurobipy, pathlib
    d = pathlib.Path(gurobipy.__file__).parent / '.libs'
    if any(d.glob('libgurobi*.so')): print(d)
except Exception: pass
" 2>/dev/null)
if [[ "$OS_NAME" != "Darwin" && -n "$PIP_GRB_DIR" && -d "$PIP_GRB_DIR" ]]; then
    GRB_PIP_LIB="$PIP_GRB_DIR"
    echo "[build] pip Gurobi found: $GRB_PIP_LIB"
fi

LINK_LIB_DIR="${GRB_PIP_LIB:-$GUROBI_HOME/lib}"

echo "[build] compiling stage2_bp_pybind.cpp -> $OUT"
CXX_BIN="${CXX:-g++}"
if [[ "$OS_NAME" == "Darwin" ]]; then
    # Python extensions are bundles on macOS.  Link libomp from the selected
    # Python environment so OpenMP pricing keeps working on Apple Clang.
    PY_PREFIX=$("$PYTHON_BIN" -c "import sys; print(sys.prefix)")
    "$CXX_BIN" -O3 -Wall -bundle -std=c++17 -fPIC -fvisibility=hidden \
        -DNDEBUG -mcpu=native -Xpreprocessor -fopenmp \
        $PY_INCLUDES \
        -I"$PY_PREFIX/include" \
        -I"$GUROBI_HOME/include" \
        stage2_bp_pybind.cpp \
        -L"$GUROBI_HOME/lib" -lgurobi_c++ -l"$GRB_LIB" \
        -L"$PY_PREFIX/lib" -lomp \
        -Wl,-rpath,"$GUROBI_HOME/lib" \
        -Wl,-rpath,"$PY_PREFIX/lib" \
        -undefined dynamic_lookup \
        -o "$OUT_TMP"
else
    "$CXX_BIN" -O3 -Wall -shared -std=c++17 -fPIC -fvisibility=hidden \
        -DNDEBUG -march=native -fopenmp \
        $PY_INCLUDES \
        -I"$GUROBI_HOME/include" \
        stage2_bp_pybind.cpp \
        -L"$GUROBI_HOME/lib" -lgurobi_c++ \
        -L"$LINK_LIB_DIR" -l"$GRB_LIB" \
        -Wl,-rpath,"$LINK_LIB_DIR" \
        -lgomp \
        -o "$OUT_TMP"
fi

mv -f "$OUT_TMP" "$OUT"
trap - EXIT
for old in stage2_bp_cpp*.so; do
    [[ -e "$old" ]] || continue
    [[ "$old" == "$OUT" ]] || rm -f "$old"
done

echo "[build] verifying import..."
"$PYTHON_BIN" -c "import sys; sys.path.insert(0, '.'); import stage2_bp_cpp; print('  module:', stage2_bp_cpp.__file__); print('  funcs:', [n for n in dir(stage2_bp_cpp) if not n.startswith('_')])"

# Build the PIP license shim if pip Gurobi was detected and the shim source exists.
if [[ -n "$GRB_PIP_LIB" && -f "grb_pip_shim.c" ]]; then
    echo "[build] compiling grb_pip_shim.so (LD_PRELOAD for PIP license)"
    gcc -shared -fPIC -o grb_pip_shim.so grb_pip_shim.c -ldl
    echo "[build] PIP shim -> $(pwd)/grb_pip_shim.so"
    echo "[build] Usage: export GRB_LICENSE_FILE=$GRB_PIP_LIB/gurobi.lic"
    echo "         export GRB_PIP_SHIM_LIBPATH=$GRB_PIP_LIB/libgurobi130.so"
    echo "         export LD_PRELOAD=$(pwd)/grb_pip_shim.so"
fi

echo "[build] OK -> $(pwd)/$OUT"
