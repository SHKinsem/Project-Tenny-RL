#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-horizon-ablation-20261116
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$HOME/projects/tenny-reward-ablation-20261116/jax-cache
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B -m experiments.reward_ablation.matrix --manifest "$work/inputs/manifest30.json" --output-dir "$work/matrix30"
