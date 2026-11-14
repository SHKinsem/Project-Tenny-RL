#!/usr/bin/env bash
# Local-only runner: reuses the user's installed ML venv, never pip-installs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${TENNY_PYTHON:-$HOME/projects/tenny-reference/.venv-gpu/bin/python}"
BASE="${TENNY_HYBRID_ROOT:-$HOME/projects/tenny-hybrid200}"
# Tests explicitly recheck; formal training opts in to reusing exact evidence.
REUSE=0
FORCE=0
TRAIN_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --reuse-ready) REUSE=1 ;;
    --force-prepare) FORCE=1 ;;
    *) TRAIN_ARGS+=("$arg") ;;
  esac
done
(( FORCE == 0 )) || REUSE=0
set -- "${TRAIN_ARGS[@]}"
test -x "$PY" || { echo "Missing ML Python: set TENNY_PYTHON to your existing environment." >&2; exit 2; }
mkdir -p "$BASE/src" "$BASE/artifacts" "$BASE/jax-cache"
# Hold through collection and all child processes: a second launcher must not
# rsync over Python sources while a first run is compiling or training.
command -v flock >/dev/null || { echo "Hybrid runner requires flock (util-linux)." >&2; exit 2; }
exec 9>"$BASE/.runner.lock"
flock -n 9 || { echo "Hybrid task already running in this runtime directory; wait for it to finish." >&2; exit 3; }
rsync -a --exclude=.git/ --exclude=__pycache__/ --exclude=results/ "$ROOT/experiments/torque200/" "$BASE/src/"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION-0.35}"
export JAX_COMPILATION_CACHE_DIR="$BASE/jax-cache"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS-1}"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH
RUN="$(date +%Y%m%d-%H%M%S)-$$"
OUT="$BASE/artifacts/$RUN"
mkdir "$OUT"
cd "$BASE/src"
collect() {
  local run_code=$? collect_code=0
  trap - EXIT
  if [[ -f "$OUT/readiness.json" ]]; then
    local evidence
    evidence="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("evidence_dir", ""))' "$OUT/readiness.json")"
    if [[ -n "$evidence" && -d "$evidence" ]]; then
      mkdir -p "$OUT/preparation"
      rsync -a --exclude=checkpoints/ --exclude=cpu_checkpoint/ "$evidence/" "$OUT/preparation/" || collect_code=$?
    fi
  fi
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
MODE="$("$PY" -c 'from hybrid.train import parser; print(parser().parse_args().mode)' \
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
echo "[$(date +%H:%M:%S)] 检查训练准备：控制频率 $CONTROL_HZ Hz，物理子步 400 Hz；首次或配置变化时完整预检，之后可复用。"
READY_ARGS=()
(( REUSE == 1 )) || READY_ARGS+=(--force)
"$PY" -u -m hybrid.readiness --runtime-root "$BASE" --mode "$MODE" \
  --control-hz "$CONTROL_HZ" --result "$OUT/readiness.json" "${READY_ARGS[@]}"
VALIDATION="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["validation_path"])' "$OUT/readiness.json")"
# Determine the caller's requested workload through the same argument parser.
STEPS="$("$PY" -c 'from hybrid.train import parser; print(parser().parse_args().steps)' \
  --validation "$OUT/validation.json" --output "$OUT/requested" "$@")"
if (( STEPS > 16 )); then
  run_logged "正式训练" "$OUT/requested.log" "$OUT/requested/report.json" \
    "$PY" -u -m hybrid.train --validation "$VALIDATION" --output "$OUT/requested" "$@"
fi
echo "[完成] 本次训练管线全部通过。16 步冒烟不代表平衡、收敛或实机通过。"
echo "HYBRID_RUN_COMPLETE $OUT"
