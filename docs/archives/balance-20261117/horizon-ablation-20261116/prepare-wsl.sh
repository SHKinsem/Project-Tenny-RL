#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-horizon-ablation-20261116
repo='${REPO}'
mkdir -p "$work/experiments/horizon_ablation" "$work/experiments/balance_diagnostics"
cp "$repo/experiments/horizon_ablation/__init__.py" "$work/experiments/horizon_ablation/"
cp "$repo/experiments/horizon_ablation/contract.py" "$work/experiments/horizon_ablation/"
rsync -a --exclude=__pycache__/ "$repo/experiments/balance_diagnostics/" "$work/experiments/balance_diagnostics/"
cp "$repo/tasks/results/horizon-ablation-20261116/prepare.py" "$work/prepare.py"
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$work/prepare.py"
# Analysis module is outside the six-file frozen training dependency set.
cp "$repo/experiments/reward_ablation/evaluation.py" "$work/experiments/reward_ablation/"
mkdir -p "$repo/tasks/results/horizon-ablation-20261116/inputs"
cp "$work/inputs/manifest30.json" "$work/inputs/reuse-verification.json" "$repo/tasks/results/horizon-ablation-20261116/inputs/"
