#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-reward-ablation-20261116
source_root='${REPO}'
mkdir -p "$work/experiments/reward_ablation" "$work/experiments/balance_diagnostics"
rsync -a --exclude=__pycache__/ "$source_root/experiments/reward_ablation/" "$work/experiments/reward_ablation/"
rsync -a --exclude=__pycache__/ "$source_root/experiments/balance_diagnostics/" "$work/experiments/balance_diagnostics/"
cp "$source_root/tasks/results/reward-ablation-20261116/prepare.py" "$work/prepare.py"
cp "$source_root/tasks/results/reward-ablation-20261116/smoke.py" "$work/smoke.py"
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$work/jax-cache"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$work/prepare.py"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$work/smoke.py"
cp "$work/smoke.json" "$source_root/tasks/results/reward-ablation-20261116/smoke.json"
