#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jaccob/projects/tenny-reference
cp '/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny/tasks/training_smoke.py' "$ROOT/training_smoke.py"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax-cache"
export OMP_NUM_THREADS=4
cd "$ROOT/mujoco_legwheel_rl_public"
RUN_ID="train-smoke-$(date +%Y%m%d-%H%M%S)"
"$ROOT/mujoco_esp32_robot_deploy_public/.venv/bin/python" -u "$ROOT/training_smoke.py" --repo "$PWD" --deploy-root "$ROOT/mujoco_esp32_robot_deploy_public" --output "$ROOT/artifacts/$RUN_ID" "$@" 2>&1 | tee "$ROOT/logs/$RUN_ID.log"
