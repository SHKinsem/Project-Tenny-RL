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
# 从最近一次 test-torque200 的 resumed checkpoint 开始
REPORT=$(ls -d "$BASE"/artifacts/smoke-*/resumed/report.json 2>/dev/null | sort | tail -n 1 || true)
test -n "$REPORT" || { echo "找不到 smoke-*/resumed/report.json，先运行 test-torque200" >&2; exit 1; }
RESTORE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$REPORT")
"$PY" -u performance.py --base "$BASE" --restore "$RESTORE" "$@"
