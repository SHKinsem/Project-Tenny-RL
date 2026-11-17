#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-reward-ablation-20261116
source_root='${REPO}'
rsync -a --exclude=__pycache__/ "$source_root/experiments/reward_ablation/" "$work/experiments/reward_ablation/"
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$work/jax-cache"
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B -m experiments.reward_ablation --manifest "$work/inputs/manifest.json" --arm baseline --seed 0 --output-dir "$work/restore-probe"
