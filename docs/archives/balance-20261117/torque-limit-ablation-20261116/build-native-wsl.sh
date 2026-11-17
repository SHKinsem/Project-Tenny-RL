#!/usr/bin/env bash
set -euo pipefail
work=$HOME/projects/tenny-torque-limit-ablation-20261116
py=$HOME/projects/tenny-reference/.venv-gpu/bin/python
export PYTHONPATH="$HOME/projects/tenny-native-eval/build-deps"
bind_dir="$($py -m pybind11 --cmakedir)"
cmake -S "$work/native/src" -B "$work/native/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$py" -Dpybind11_DIR="$bind_dir" -DNATIVE_EVAL_OUTPUT_DIR="$work/native/python/native_eval"
cmake --build "$work/native/build" --parallel 4
