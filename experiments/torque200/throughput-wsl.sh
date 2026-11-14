#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TENNY_ROOT="$WINROOT"
set -euo pipefail
BASE=$HOME/projects/tenny-torque200
WIN="$WINROOT/experiments/torque200"
PY=$HOME/projects/tenny-reference/.venv-gpu/bin/python
cp "$WIN/throughput_probe.py" "$BASE/src/"
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="$BASE/jax-cache"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
unset LD_LIBRARY_PATH
collect() {
    for run in "$BASE"/artifacts/throughput-20*; do
        test -d "$run" || continue
        mkdir -p "$WIN/results/$(basename "$run")"
        rsync -a --exclude=checkpoints/ --exclude=cpu_checkpoint/ "$run/" "$WIN/results/$(basename "$run")/"
    done
}
trap collect EXIT
cd "$BASE/src"
"$PY" -u throughput_probe.py
