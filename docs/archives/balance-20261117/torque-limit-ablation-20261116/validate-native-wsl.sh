#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-torque-limit-ablation-20261116
repo='${REPO}'
export PYTHONPATH="$work/native/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/torque-limit-ablation-20261116/validate_native.py"
