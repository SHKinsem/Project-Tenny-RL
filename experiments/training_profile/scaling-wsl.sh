#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TENNY_SCALING_WORKDIR:-$HOME/projects/tenny-parallel-scaling}"
PY="${TENNY_PYTHON:-$HOME/projects/tenny-reference/.venv-gpu/bin/python}"
WORK="$(realpath -m -- "$WORK")"
case "$WORK" in /mnt/*|/|"") printf '%s\n' 'Use a dedicated WSL ext4 work directory' >&2; exit 2;; esac
test -x "$PY"
mkdir -p "$WORK/src"
rsync -a --exclude=__pycache__/ --exclude=.pytest_cache/ --exclude=results/ "$SOURCE/" "$WORK/src/"
exec "$PY" -B "$WORK/src/launch_scaling.py" "$@"
