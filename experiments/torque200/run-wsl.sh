#!/usr/bin/env bash
set -euo pipefail
BASE=/home/jaccob/projects/tenny-torque200
ORIGINAL=/home/jaccob/projects/tenny-reference
WIN='/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny/experiments/torque200'
PY="$ORIGINAL/.venv-gpu/bin/python"
test -x "$PY"
test -d "$ORIGINAL/mujoco_legwheel_rl_public"
test -d "$ORIGINAL/mujoco_esp32_robot_deploy_public"
mkdir -p "$BASE/src" "$BASE/artifacts" "$BASE/jax-cache"
# Independent source on ext4; do not sync back into either original repository.
rsync -a --exclude=.git/ --exclude=.codegraph/ --exclude=__pycache__/ --exclude=results/ "$WIN/" "$BASE/src/"
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="$BASE/jax-cache"
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
unset LD_LIBRARY_PATH
RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"
OUT="$BASE/artifacts/$RUN_ID"
mkdir "$OUT"
collect() {
    mkdir -p "$WIN/results/$RUN_ID"
    rsync -a --exclude=checkpoints/ --exclude=cpu_checkpoint/ "$OUT/" "$WIN/results/$RUN_ID/"
}
trap collect EXIT
cd "$BASE/src"
"$PY" preserve_original.py snapshot "$OUT/original-before.json"
"$PY" -m compileall -q legwheel smoke.py tests/test_env.py
"$PY" -u tests/test_env.py 2>&1 | tee "$OUT/environment-tests.log"
"$PY" -u smoke.py --repo "$BASE/src" --expected-platform gpu --output "$OUT/first" "$@" 2>&1 | tee "$OUT/first.log"
RESTORE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUT/first/report.json")
"$PY" -u smoke.py --repo "$BASE/src" --expected-platform gpu --output "$OUT/resumed" --restore "$RESTORE" "$@" 2>&1 | tee "$OUT/resumed.log"
"$PY" verify_artifacts.py "$OUT/first" "$OUT/resumed" | tee "$OUT/parameter-verification.log"
"$PY" preserve_original.py verify "$OUT/original-before.json" | tee "$OUT/original-preserved.log"
echo "TORQUE200_ALL_CHECKS_OK $OUT"
