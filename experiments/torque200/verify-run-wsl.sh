#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TENNY_ROOT="$WINROOT"
# Recheck a completed local run without performing more training.
set -euo pipefail
BASE=/home/jaccob/projects/tenny-torque200
WIN="$WINROOT/experiments/torque200"
PY=/home/jaccob/projects/tenny-reference/.venv-gpu/bin/python
RUN_ID="${1:?Pass a smoke-YYYYMMDD-HHMMSS run ID}"
[[ "$RUN_ID" =~ ^smoke-[0-9]{8}-[0-9]{6}$ ]]
OUT="$BASE/artifacts/$RUN_ID"
test -f "$OUT/resumed/report.json"
# Both training processes have exited before this read-only parameter validation.
cp "$WIN/verify_artifacts.py" "$BASE/src/verify_artifacts.py"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
cd "$BASE/src"
"$PY" verify_artifacts.py "$OUT/first" "$OUT/resumed" | tee "$OUT/parameter-verification.log"
"$PY" preserve_original.py verify "$OUT/original-before.json" | tee "$OUT/original-preserved.log"
mkdir -p "$WIN/results/$RUN_ID"
rsync -a --exclude=checkpoints/ --exclude=cpu_checkpoint/ "$OUT/" "$WIN/results/$RUN_ID/"
