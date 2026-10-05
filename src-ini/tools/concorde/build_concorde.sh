#!/usr/bin/env bash
# Build Concorde (co031219) against QSopt and install the binary next to this
# script as tools/concorde/concorde, which solvers/forward_stage3_solver.py
# resolves automatically (or point VRP_CONCORDE_BIN at it).
#
# macOS notes:
#   * QSopt is only distributed as an x86_64 archive, so on Apple Silicon the
#     binary is built for x86_64 and runs through Rosetta 2
#     (softwareupdate --install-rosetta once, if missing).
#   * Concorde's Makefiles cannot cope with spaces in paths, so the build runs
#     in a scratch directory outside the (iCloud) repository.
#   * Downloaded archives are cached in tools/concorde_build/ (git-ignored);
#     delete them to force a fresh download.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${HERE}/../concorde_build"
WORK="${CONCORDE_BUILD_DIR:-/tmp/concorde_build}"
SRC_URL="https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz"

case "$(uname -s)" in
  Darwin)
    QS_URL="https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64"
    ARCH_FLAG="-arch x86_64"
    HOST_TRIPLE="x86_64-apple-darwin"
    ;;
  Linux)
    QS_URL="https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/linux64"
    ARCH_FLAG=""
    HOST_TRIPLE=""
    ;;
  *)
    echo "unsupported platform: $(uname -s)" >&2
    exit 1
    ;;
esac

mkdir -p "${CACHE}/qsopt"
[ -f "${CACHE}/co031219.tgz" ] || curl -fsSL -o "${CACHE}/co031219.tgz" "${SRC_URL}"
[ -f "${CACHE}/qsopt/qsopt.a" ] || curl -fsSL -o "${CACHE}/qsopt/qsopt.a" "${QS_URL}/qsopt.a"
[ -f "${CACHE}/qsopt/qsopt.h" ] || curl -fsSL -o "${CACHE}/qsopt/qsopt.h" "${QS_URL}/qsopt.h"

rm -rf "${WORK}"
mkdir -p "${WORK}/concorde-bld"
cp -R "${CACHE}/qsopt" "${WORK}/qsopt"
tar xzf "${CACHE}/co031219.tgz" -C "${WORK}"

cd "${WORK}/concorde-bld"
export CC="cc ${ARCH_FLAG}"
# The 2003 sources rely on K&R-era C that modern clang/gcc reject by default.
export CFLAGS="-O2 -Wno-implicit-function-declaration -Wno-int-conversion \
  -Wno-incompatible-function-pointer-types -Wno-implicit-int \
  -Wno-deprecated-non-prototype -Wno-return-type -Wno-error"
CONFIG_ARGS=(--with-qsopt="${WORK}/qsopt")
if [ -n "${HOST_TRIPLE}" ]; then
  CONFIG_ARGS+=(--host="${HOST_TRIPLE}" --build="${HOST_TRIPLE}")
fi
../concorde/configure "${CONFIG_ARGS[@]}"
make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

install -m 755 TSP/concorde "${HERE}/concorde"
echo "installed: ${HERE}/concorde"
file "${HERE}/concorde"
