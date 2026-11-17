#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-wheel-hold-ablation-20261116
repo='${REPO}'
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$work/jax-cache"
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B -m experiments.wheel_hold_ablation.matrix --inputs "$work/inputs" --output "$work/matrix"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/wheel-hold-ablation-20261116/evaluate.py"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/wheel-hold-ablation-20261116/verify.py"
