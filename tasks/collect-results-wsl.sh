#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TENNY_ROOT="$WINROOT"
set -euo pipefail
ROOT=$HOME/projects/tenny-reference
OUT="$WINROOT/tasks/results"
mkdir -p "$OUT"
cp "$ROOT/logs/requirements-resolved.txt" "$ROOT/logs/unit-tests.log" "$ROOT/logs/sim-verify.log" "$OUT/"
cp "$ROOT/mujoco_esp32_robot_deploy_public/tests/sim_verify_hold_obs_results.json" "$OUT/"
for run in "$ROOT"/artifacts/train-smoke-*; do
  [ -f "$run/report.json" ] || continue
  mkdir -p "$OUT/$(basename "$run")"
  cp "$run/report.json" "$run/ppo_config.json" "$run/env_config.json" "$run/progress.json" "$run/smoke_params.cpp.txt" "$OUT/$(basename "$run")/"
done
