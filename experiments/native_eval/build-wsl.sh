#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK="${NATIVE_EVAL_WORKDIR:-$HOME/projects/tenny-native-eval}"
PY="${TENNY_PYTHON:-$HOME/projects/tenny-reference/.venv-gpu/bin/python}"
test -x "$PY"
mkdir -p "$WORK/src" "$WORK/python" "$WORK/build-deps"
rsync -a --exclude=.git/ --exclude=.codegraph/ --exclude=__pycache__/ --exclude=results/ \
  --exclude=build/ "$SOURCE/" "$WORK/src/"
if ! "$PY" -c 'import pybind11' >/dev/null 2>&1; then
  if ! PYTHONPATH="$WORK/build-deps${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c 'import pybind11' >/dev/null 2>&1; then
    "$PY" -m pip install --disable-pip-version-check --no-deps --target "$WORK/build-deps" 'pybind11==3.0.1'
  fi
  export PYTHONPATH="$WORK/build-deps${PYTHONPATH:+:$PYTHONPATH}"
fi
PYBIND_DIR="$("$PY" -m pybind11 --cmakedir)"
rsync -a --exclude=__pycache__/ "$WORK/src/python/" "$WORK/python/"
cmake -S "$WORK/src" -B "$WORK/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$PYBIND_DIR" -DNATIVE_EVAL_OUTPUT_DIR="$WORK/python/native_eval"
cmake --build "$WORK/build" --parallel 4
printf 'NATIVE_EVAL_PYTHONPATH=%s\n' "$WORK/python"
printf 'NATIVE_EVAL_TEST_ROOT=%s\n' "$WORK/src/tests"
