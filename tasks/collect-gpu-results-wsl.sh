#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TENNY_ROOT="$WINROOT"
set -euo pipefail
shopt -s nullglob
ROOT=/home/jaccob/projects/tenny-reference
OUT="$WINROOT/tasks/results/gpu"
mkdir -p "$OUT"
for file in "$ROOT"/artifacts/gpu-*.json "$ROOT"/artifacts/benchmark-*.json "$ROOT"/logs/*gpu*.csv "$ROOT"/logs/gpu-*.log "$ROOT"/logs/train-gpu-*.log "$ROOT"/logs/benchmark-*.log "$ROOT"/logs/requirements-gpu-resolved.txt "$ROOT"/logs/install-cuda12.log; do
  [ -f "$file" ] && cp "$file" "$OUT/"
done
for run in "$ROOT"/artifacts/train-gpu-*; do
  [ -d "$run" ] || continue
  mkdir -p "$OUT/$(basename "$run")"
  for name in report.json ppo_config.json env_config.json progress.json smoke_params.cpp.txt; do
    [ ! -f "$run/$name" ] || cp "$run/$name" "$OUT/$(basename "$run")/"
  done
done
