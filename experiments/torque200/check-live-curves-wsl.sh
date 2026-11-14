#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TENNY_ROOT="$WINROOT"
set -euo pipefail
BASE=$HOME/projects/tenny-torque200
WIN="$WINROOT/experiments/torque200"
mkdir -p "$BASE/src/tests"
cp "$WIN/tests/test_live_curves.py" "$BASE/src/tests/"
cd "$BASE/src"
$HOME/projects/tenny-reference/.venv-gpu/bin/python -m unittest discover -s tests -p test_live_curves.py -v
