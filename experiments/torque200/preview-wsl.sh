#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TENNY_ROOT="$WINROOT"
set -euo pipefail
BASE=/home/jaccob/projects/tenny-torque200
WIN="$WINROOT/experiments/torque200"
PY=/home/jaccob/projects/tenny-reference/.venv-gpu/bin/python
test -x "$PY"
test -d "$BASE/src/legwheel"
# Copy only the display entry. Training source and original project stay intact.
cp "$WIN/preview.py" "$BASE/src/preview.py"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$BASE/jax-cache"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
unset MUJOCO_GL
OUT="$BASE/artifacts/preview-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
collect() {
    mkdir -p "$WIN/results/$(basename "$OUT")"
    cp "$OUT/"* "$WIN/results/$(basename "$OUT")/"
}
trap collect EXIT
cd "$BASE/src"
"$PY" -u preview.py --artifacts "$BASE/artifacts" --output "$OUT" "$@" 2>&1 | tee "$OUT/preview.log"
