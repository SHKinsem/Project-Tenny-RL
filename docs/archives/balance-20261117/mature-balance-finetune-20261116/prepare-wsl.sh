#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-mature-finetune-20261116
repo='${REPO}'
mkdir -p "$work/experiments"
for package in mature_balance_finetune reward_ablation balance_diagnostics; do
    mkdir -p "$work/experiments/$package"
    rsync -a --exclude=__pycache__/ "$repo/experiments/$package/" "$work/experiments/$package/"
done
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/mature-balance-finetune-20261116/prepare.py"
