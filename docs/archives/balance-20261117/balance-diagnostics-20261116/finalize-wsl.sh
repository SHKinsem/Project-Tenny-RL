#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-balance-diagnostics-20261116
source_root='${REPO}'
rsync -a --exclude=__pycache__/ "$source_root/experiments/balance_diagnostics/" "$work/experiments/balance_diagnostics/"
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$work"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$work/matplotlib"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B -m experiments.balance_diagnostics report "$work/result/results.json" --output-dir "$work/result"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$source_root/tasks/results/balance-diagnostics-20261116/verify_inputs.py"
rsync -a "$work/result/" "$source_root/tasks/results/balance-diagnostics-20261116/result/"
