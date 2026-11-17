#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-mature-finetune-20261116
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$work/jax-cache"
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B -m experiments.mature_balance_finetune --manifest "$work/inputs/smoke.json" --seed 0 --output-dir "$work/smoke"
