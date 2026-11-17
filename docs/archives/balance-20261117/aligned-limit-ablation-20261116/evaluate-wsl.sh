#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-aligned-limit-ablation-20261116
repo='${REPO}'
export PYTHONPATH="$work/native/python:$HOME/projects/tenny-torque200/src:$work"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$work/matplotlib"
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/aligned-limit-ablation-20261116/evaluate.py"
