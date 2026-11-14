#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TENNY_ROOT="$WINROOT"
set -euo pipefail
BASE=/home/jaccob/projects/tenny-torque200
WIN="$WINROOT/experiments/torque200"
PY=/home/jaccob/projects/tenny-reference/.venv-gpu/bin/python
test -x "$PY"
cp "$WIN/timed_train.py" "$BASE/src/timed_train.py"
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
    for run in "$BASE"/artifacts/training-20* "$BASE"/artifacts/timed-validation-20*; do
        test -d "$run" || continue
        mkdir -p "$WIN/results/$(basename "$run")"
        rsync -a --exclude=checkpoints/ --exclude=cpu_checkpoint/ "$run/" "$WIN/results/$(basename "$run")/"
    done
    for name in training-live.json training-status.json training-latest.json; do
        test ! -f "$BASE/artifacts/$name" || cp "$BASE/artifacts/$name" "$WIN/results/$name"
    done
}
trap collect EXIT
cd "$BASE/src"
"$PY" -u timed_train.py --base "$BASE" "$@"
