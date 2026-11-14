#!/usr/bin/env bash
set -euo pipefail
EXPECTED=/home/jaccob/projects/tenny-reference/.venv-gpu/bin/python
if test -r /proc/9853/cmdline; then
    /usr/bin/python3 - <<'PY'
from pathlib import Path
p = Path('/proc/9853/cmdline')
args = p.read_bytes().split(b'\0')
assert args[0] == b'/home/jaccob/projects/tenny-reference/.venv-gpu/bin/python', args
assert b'web_preview.py' in args, args
assert b'/home/jaccob/projects/tenny-torque200/artifacts' in args, args
PY
    kill -INT 9853
    for attempt in {1..40}; do
        test ! -d /proc/9853 && break
        sleep 0.1
    done
fi
exec bash './experiments/torque200/web-preview-wsl.sh' --port 8770 --mode original
