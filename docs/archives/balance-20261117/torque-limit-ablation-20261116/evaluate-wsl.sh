#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-torque-limit-ablation-20261116
repo='${REPO}'
cp "$repo/experiments/balance_diagnostics/runner.py" "$work/experiments/balance_diagnostics/runner.py"
export PYTHONPATH="$work/native/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$work/matplotlib"
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/torque-limit-ablation-20261116/evaluate.py"
