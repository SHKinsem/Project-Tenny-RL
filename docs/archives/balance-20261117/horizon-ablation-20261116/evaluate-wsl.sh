#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-horizon-ablation-20261116
repo='${REPO}'
cp "$repo/experiments/horizon_ablation/report.py" "$work/experiments/horizon_ablation/"
cp "$repo/experiments/reward_ablation/report.py" "$work/experiments/reward_ablation/report.py"
cp "$repo/tasks/results/horizon-ablation-20261116/evaluate.py" "$work/evaluate.py"
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$work/matplotlib"
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$work/evaluate.py"
