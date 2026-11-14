#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jaccob/projects/tenny-reference
cd "$ROOT/mujoco_esp32_robot_deploy_public"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax-cache"
export OMP_NUM_THREADS=4
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > "$ROOT/logs/requirements-resolved.txt"
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v 2>&1 | tee "$ROOT/logs/unit-tests.log"
.venv/bin/python -u tests/sim_verify_hold_obs.py 2>&1 | tee "$ROOT/logs/sim-verify.log"
