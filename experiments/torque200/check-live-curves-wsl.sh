#!/usr/bin/env bash
set -euo pipefail
BASE=/home/jaccob/projects/tenny-torque200
WIN='/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny/experiments/torque200'
mkdir -p "$BASE/src/tests"
cp "$WIN/tests/test_live_curves.py" "$BASE/src/tests/"
cd "$BASE/src"
/home/jaccob/projects/tenny-reference/.venv-gpu/bin/python -m unittest discover -s tests -p test_live_curves.py -v
