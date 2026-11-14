#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jaccob/projects/tenny-reference
cd "$ROOT/mujoco_esp32_robot_deploy_public"
export SIM_NO_BROWSER=1
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax-cache"
export OMP_NUM_THREADS=4
# Call the original entry directly: sim_web.sh kills any existing port owner.
SIM_HOST=$(hostname -I)
SIM_HOST=${SIM_HOST%% *}
echo "Windows browser: http://$SIM_HOST:8765/"
exec .venv/bin/python -u tests/sim_web_server.py --host "$SIM_HOST" --port 8765 --jax-platforms cpu --noise false --random-init false --hold-clip-wheel 3.0 --hold-clip-sinyaw 1.0
