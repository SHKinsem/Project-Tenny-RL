#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-reward-ablation-20261116
source_root='${REPO}'
# Only copy analysis modules; do not touch the frozen training snapshot.
cp "$source_root/experiments/reward_ablation/evaluation.py" "$work/experiments/reward_ablation/evaluation.py"
cp "$source_root/experiments/reward_ablation/report.py" "$work/experiments/reward_ablation/report.py"
cp "$source_root/tasks/results/reward-ablation-20261116/evaluate_available.py" "$work/evaluate_available.py"
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$work/matplotlib"
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$work/evaluate_available.py"
mkdir -p "$source_root/tasks/results/reward-ablation-20261116/comparison"
rsync -a "$work/comparison/" "$source_root/tasks/results/reward-ablation-20261116/comparison/"
cp "$work/matrix-v2/status.json" "$source_root/tasks/results/reward-ablation-20261116/matrix-status.json"
cp "$work/inputs/manifest.json" "$source_root/tasks/results/reward-ablation-20261116/manifest.json"
