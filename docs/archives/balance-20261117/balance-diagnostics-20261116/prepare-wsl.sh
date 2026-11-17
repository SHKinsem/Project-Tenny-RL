#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-balance-diagnostics-20261116
source_root='${REPO}'
mkdir -p "$work/experiments/balance_diagnostics"
rsync -a --exclude=__pycache__/ "$source_root/experiments/balance_diagnostics/" "$work/experiments/balance_diagnostics/"
cp "$source_root/tasks/results/balance-diagnostics-20261116/prepare_inputs.py" "$work/prepare_inputs.py"
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$work/prepare_inputs.py"
