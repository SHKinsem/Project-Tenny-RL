#!/usr/bin/env bash
WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TENNY_ROOT="$WINROOT"
set -euo pipefail
ROOT=$HOME/projects/tenny-reference
SOURCE="$WINROOT/reference/"
mkdir -p "$ROOT" "$ROOT/logs"
rsync -a --exclude=.pio/ --exclude=.codegraph/ "$SOURCE" "$ROOT/"
cd "$ROOT/mujoco_esp32_robot_deploy_public"
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install -r requirements-sim.txt 2>&1 | tee "$ROOT/logs/install-sim.log"
