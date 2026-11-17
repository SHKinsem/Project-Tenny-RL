#!/usr/bin/env bash
set -euo pipefail
repo='${REPO}'
export PYTHONPATH="$repo:$HOME/projects/tenny-torque200/src:$HOME/projects/tenny-native-eval/python"
export PYTHONDONTWRITEBYTECODE=1
cd $HOME/projects
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/torque-limit-audit-20261116/run.py"
