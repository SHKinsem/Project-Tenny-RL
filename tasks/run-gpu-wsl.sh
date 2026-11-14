#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jaccob/projects/tenny-reference
TASKS='/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny/tasks'
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax-cache-gpu"
export OMP_NUM_THREADS=4
unset LD_LIBRARY_PATH
cp "$TASKS/gpu-probe.py" "$ROOT/gpu-probe.py"
"$ROOT/.venv-gpu/bin/python" "$ROOT/gpu-probe.py" 2>&1 | tee "$ROOT/logs/gpu-probe.log"
cp "$TASKS/training_smoke.py" "$ROOT/training_smoke.py"
cd "$ROOT/mujoco_legwheel_rl_public"
RUN_ID="train-gpu-$(date +%Y%m%d-%H%M%S)"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > "$ROOT/logs/$RUN_ID-gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
"$ROOT/.venv-gpu/bin/python" -u "$ROOT/training_smoke.py" --repo "$PWD" --deploy-root "$ROOT/mujoco_esp32_robot_deploy_public" --expected-platform gpu --output "$ROOT/artifacts/$RUN_ID" "$@" 2>&1 | tee "$ROOT/logs/$RUN_ID.log"
