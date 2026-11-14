#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jaccob/projects/tenny-reference
cp '/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny/tasks/gpu-sim-verify.py' "$ROOT/gpu-sim-verify.py"
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax-cache-gpu"
export OMP_NUM_THREADS=4
unset LD_LIBRARY_PATH
"$ROOT/.venv-gpu/bin/python" -u "$ROOT/gpu-sim-verify.py" 2>&1 | tee "$ROOT/logs/gpu-sim-verify.log"
