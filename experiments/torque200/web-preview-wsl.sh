#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TENNY_ROOT="$WINROOT"
set -euo pipefail
BASE=$HOME/projects/tenny-torque200
WIN="$WINROOT/experiments/torque200"
PY=$HOME/projects/tenny-reference/.venv-gpu/bin/python
test -x "$PY"
test -d "$BASE/src/legwheel"
HYBRID_BASE="${TENNY_HYBRID_ROOT:-$HOME/projects/tenny-hybrid200}"
mkdir -p "$HYBRID_BASE"
# The same locks are held by CLI / managed training. Never replace source under
# an active run; check the persistent supervisor too for older unlocked jobs.
exec 8>"$BASE/.runner.lock"
flock -n 8 || { echo "Training is active; wait for saving before updating the console." >&2; exit 3; }
if [[ "$(realpath "$BASE")" != "$(realpath "$HYBRID_BASE")" ]]; then
  exec 9>"$HYBRID_BASE/.runner.lock"
  flock -n 9 || { echo "Hybrid training is active; wait before updating the console." >&2; exit 3; }
fi
"$PY" - "$WIN" "$BASE" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from training_control import TrainingController
base = Path(sys.argv[2])
controller = TrainingController(base / "artifacts", Path.home() / "projects/tenny-reference", base)
if controller.status().get("active"):
    raise SystemExit("Training is active; stop and save before updating the console.")
PY
rsync -a --exclude=.git/ --exclude=__pycache__/ --exclude=results/ "$WIN/" "$BASE/src/"
flock -u 8
exec 8>&-
if [[ "$(realpath "$BASE")" != "$(realpath "$HYBRID_BASE")" ]]; then
  flock -u 9
  exec 9>&-
fi
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$BASE/jax-cache"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
unset MUJOCO_GL
cd "$BASE/src"
exec "$PY" -u web_preview.py --host 0.0.0.0 --artifacts "$BASE/artifacts" "$@"
