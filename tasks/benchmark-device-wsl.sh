#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jaccob/projects/tenny-reference
BACKEND=${1:?cpu or cuda}
case "$BACKEND" in
  cpu) PYTHON="$ROOT/mujoco_esp32_robot_deploy_public/.venv/bin/python" ;;
  cuda) PYTHON="$ROOT/.venv-gpu/bin/python" ;;
  *) exit 2 ;;
esac
shift
cp '/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny/tasks/device_benchmark.py' "$ROOT/device_benchmark.py"
export JAX_PLATFORMS="$BACKEND"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax-cache-$BACKEND"
export OMP_NUM_THREADS=4
unset LD_LIBRARY_PATH
RUN_ID="benchmark-$BACKEND-$(date +%Y%m%d-%H%M%S)"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > "$ROOT/logs/$RUN_ID-gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
"$PYTHON" -u "$ROOT/device_benchmark.py" --repo "$ROOT/mujoco_legwheel_rl_public" --expected-backend "$BACKEND" --output "$ROOT/artifacts/$RUN_ID.json" "$@" 2>&1 | tee "$ROOT/logs/$RUN_ID.log"
