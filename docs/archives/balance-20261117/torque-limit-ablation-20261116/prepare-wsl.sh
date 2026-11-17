#!/usr/bin/env bash
set -euo pipefail
repo='${REPO}'
export PYTHONPATH="$repo"
export PYTHONDONTWRITEBYTECODE=1
cd $HOME/projects
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/torque-limit-ablation-20261116/prepare.py"
