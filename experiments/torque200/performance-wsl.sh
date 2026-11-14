#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TENNY_ROOT="$WINROOT"
set -euo pipefail
BASE=$HOME/projects/tenny-torque200
WIN="$WINROOT/experiments/torque200"
PY=$HOME/projects/tenny-reference/.venv-gpu/bin/python
test -x "$PY"
cp "$WIN/performance.py" "$WIN/performance_train.py" "$BASE/src/"
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="$BASE/jax-cache"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
unset LD_LIBRARY_PATH
collect() {
    for run in "$BASE"/artifacts/performance-20*; do
        test -d "$run" || continue
        mkdir -p "$WIN/results/$(basename "$run")"
        rsync -a --exclude=checkpoints/ --exclude=cpu_checkpoint/ "$run/" "$WIN/results/$(basename "$run")/"
    done
    for file in performance-latest.json performance-status.json; do
        test ! -f "$BASE/artifacts/$file" || cp "$BASE/artifacts/$file" "$WIN/results/$file"
    done
}
trap collect EXIT
cd "$BASE/src"
"$PY" -m py_compile performance.py performance_train.py
"$PY" -u performance.py --base "$BASE" --restore "$BASE/artifacts/smoke-20261114-154038/resumed/checkpoints/000000000016" "$@"
