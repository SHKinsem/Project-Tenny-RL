#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-aligned-limit-ablation-20261116
repo='${REPO}'
export PYTHONPATH="$work/native/python:$HOME/projects/tenny-torque200/src:$work"
export TENNY_ALIGNED_ANCHOR_CHECKPOINT=$HOME/projects/tenny-torque200/artifacts/force-teacher-pretrain/verified-bootstrap/d9772a5a5cf30d5c8850b13226fe075f0c494c1017138a745b6936a882227990/verification-ee717d86609f44afa0800e0e2b79270d/checkpoint
export TENNY_ALIGNED_ANCHOR_DESCRIPTOR="$work/inputs/anchor-bc.json"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$work/jax-cache"
cd "$work"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -B "$repo/tasks/results/aligned-limit-ablation-20261116/smoke.py"
