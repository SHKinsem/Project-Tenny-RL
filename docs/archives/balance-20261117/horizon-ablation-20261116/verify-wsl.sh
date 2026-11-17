#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-horizon-ablation-20261116
repo='${REPO}'
export PYTHONPATH="$HOME/projects/tenny-native-eval/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/horizon-ablation-20261116/verify.py"
