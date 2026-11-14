#!/usr/bin/env bash
# Local-only runner: reuses the user's installed ML venv, never pip-installs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${TENNY_PYTHON:-$HOME/projects/tenny-reference/.venv-gpu/bin/python}"
BASE="${TENNY_HYBRID_ROOT:-$HOME/projects/tenny-hybrid200}"
test -x "$PY" || { echo "Missing ML Python: set TENNY_PYTHON to your existing environment." >&2; exit 2; }
mkdir -p "$BASE/src" "$BASE/artifacts" "$BASE/jax-cache"
# Hold through collection and all child processes: a second launcher must not
# rsync over Python sources while a first run is compiling or training.
command -v flock >/dev/null || { echo "Hybrid runner requires flock (util-linux)." >&2; exit 2; }
exec 9>"$BASE/.runner.lock"
flock -n 9 || { echo "Hybrid task already running in this runtime directory; wait for it to finish." >&2; exit 3; }
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
  local run_code=$? collect_code=0
  trap - EXIT
  if mkdir -p "$ROOT/experiments/torque200/results/hybrid-$RUN" &&
      rsync -a --exclude=checkpoints/ --exclude=cpu_checkpoint/ "$OUT/" "$ROOT/experiments/torque200/results/hybrid-$RUN/"; then
    :
  else
    collect_code=$?
    echo "Result collection failed (exit $collect_code); original logs remain in $OUT" >&2
  fi
  # Never replace a training/validation failure with a secondary copy error.
  # A successful run with failed collection must still report nonzero overall.
  if (( run_code != 0 )); then exit "$run_code"; fi
  exit "$collect_code"
}
trap collect EXIT
# Parse once before any ML child starts. The same parser checks the request and
# routes the selected timing to preflight as well as every training phase.
CONTROL_HZ="$("$PY" -c 'from hybrid.train import parser; print(parser().parse_args().control_hz)' \
  --validation "$OUT/validation.json" --output "$OUT/requested" "$@")"
run_logged() {
  local label="$1" logfile="$2" report="$3" code
  shift 3
  local options=()
  [[ "${TENNY_VERBOSE:-0}" == "1" ]] && options+=(--verbose)
  echo "[$(date +%H:%M:%S)] $label | 完整日志：$logfile"
  if "$@" 2>&1 | "$PY" -u -m hybrid.console capture --label "$label" --log "$logfile" "${options[@]}"; then
    if "$PY" -m hybrid.console summary --label "$label" --log "$logfile" --report "$report"; then
      return 0
    else
      code=$?
    fi
  else
    code=$?
  fi
  "$PY" -m hybrid.console failure --label "$label" --log "$logfile" --code "$code" || true
  return "$code"
}
echo "[$(date +%H:%M:%S)] 开始预检：控制频率 $CONTROL_HZ Hz，物理子步 400 Hz；首次 JAX/XLA 编译可能耗时数分钟；每 15 秒输出当前阶段等待心跳，不估算剩余时间。"
run_logged "预检" "$OUT/validation.log" "$OUT/validation.json" \
  "$PY" -u -m hybrid.validate --output "$OUT/validation.json" --control-hz "$CONTROL_HZ"
echo "[$(date +%H:%M:%S)] 预检通过；开始 16 步新训练，编译和训练时间计入实际耗时。"
# Every request first proves a tiny update, save/reload/export, then a warm
# restart with changed network weights. No successful report => no long run.
run_logged "新训练" "$OUT/train.log" "$OUT/first/report.json" \
  "$PY" -u -m hybrid.train --validation "$OUT/validation.json" --output "$OUT/first" "$@" \
  --steps 16 --num-envs 2 --eval-envs 2 --unroll-length 4
echo "[$(date +%H:%M:%S)] 新训练检查通过；准备 16 步检查点热启动。"
CHECKPOINT="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUT/first/report.json")"
run_logged "热启动" "$OUT/restart.log" "$OUT/restart/report.json" \
  "$PY" -u -m hybrid.train --validation "$OUT/validation.json" --output "$OUT/restart" "$@" \
  --steps 16 --num-envs 2 --eval-envs 2 --unroll-length 4 --restore "$CHECKPOINT"
# Determine the caller's requested workload through the same argument parser.
STEPS="$("$PY" -c 'from hybrid.train import parser; print(parser().parse_args().steps)' \
  --validation "$OUT/validation.json" --output "$OUT/requested" "$@")"
if (( STEPS > 16 )); then
  run_logged "正式训练" "$OUT/requested.log" "$OUT/requested/report.json" \
    "$PY" -u -m hybrid.train --validation "$OUT/validation.json" --output "$OUT/requested" "$@"
fi
echo "[完成] 本次训练管线全部通过。16 步冒烟不代表平衡、收敛或实机通过。"
echo "HYBRID_RUN_COMPLETE $OUT"
