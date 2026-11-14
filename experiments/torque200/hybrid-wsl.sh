#!/usr/bin/env bash
# Local-only runner: reuses the user's installed ML venv, never pip-installs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${TENNY_PYTHON:-$HOME/projects/tenny-reference/.venv-gpu/bin/python}"
BASE="${TENNY_HYBRID_ROOT:-$HOME/projects/tenny-hybrid200}"
test -x "$PY" || { echo "Missing ML Python: set TENNY_PYTHON to your existing environment." >&2; exit 2; }
mkdir -p "$BASE/src" "$BASE/artifacts" "$BASE/jax-cache"
rsync -a --exclude=.git/ --exclude=__pycache__/ --exclude=results/ "$ROOT/experiments/torque200/" "$BASE/src/"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$BASE/jax-cache"
export MPLBACKEND=Agg
RUN="$(date +%Y%m%d-%H%M%S)-$$"
OUT="$BASE/artifacts/$RUN"
mkdir "$OUT"
cd "$BASE/src"
collect() {
  mkdir -p "$ROOT/experiments/torque200/results/hybrid-$RUN"
  rsync -a --exclude=checkpoints/ --exclude=cpu_checkpoint/ "$OUT/" "$ROOT/experiments/torque200/results/hybrid-$RUN/"
}
trap collect EXIT
echo "[$(date +%H:%M:%S)] 开始预检：首次 JAX/XLA 编译可能耗时数分钟；每 15 秒输出当前阶段等待心跳，不估算剩余时间。"
"$PY" -u -m hybrid.validate --output "$OUT/validation.json" 2>&1 | tee "$OUT/validation.log"
echo "[$(date +%H:%M:%S)] 预检通过；开始 16 步新训练，编译和训练时间计入实际耗时。"
# Every request first proves a tiny update, save/reload/export, then a warm
# restart with changed network weights. No successful report => no long run.
"$PY" -u -m hybrid.train --validation "$OUT/validation.json" --output "$OUT/first" "$@" \
  --steps 16 --num-envs 2 --eval-envs 2 --unroll-length 4 2>&1 | tee "$OUT/train.log"
echo "[$(date +%H:%M:%S)] 新训练检查通过；准备 16 步检查点热启动。"
CHECKPOINT="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUT/first/report.json")"
"$PY" -u -m hybrid.train --validation "$OUT/validation.json" --output "$OUT/restart" "$@" \
  --steps 16 --num-envs 2 --eval-envs 2 --unroll-length 4 --restore "$CHECKPOINT" 2>&1 | tee "$OUT/restart.log"
# Determine the caller's requested workload through the same argument parser.
STEPS="$("$PY" -c 'from hybrid.train import parser; print(parser().parse_args().steps)' \
  --validation "$OUT/validation.json" --output "$OUT/requested" "$@")"
if (( STEPS > 16 )); then
  "$PY" -u -m hybrid.train --validation "$OUT/validation.json" --output "$OUT/requested" "$@" 2>&1 | tee "$OUT/requested.log"
fi
echo "HYBRID_RUN_COMPLETE $OUT"
