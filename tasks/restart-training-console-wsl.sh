#!/usr/bin/env bash
set -euo pipefail
/usr/bin/python3 - <<'PY'
import os, signal
from pathlib import Path
pid=15585
path=Path(f'/proc/{pid}/cmdline')
if path.exists():
    args=path.read_bytes().split(b'\0')
    assert args[0] == b'/home/jaccob/projects/tenny-reference/.venv-gpu/bin/python'
    assert any(x.endswith(b'web_preview.py') for x in args)
    assert b'/home/jaccob/projects/tenny-torque200/artifacts' in args
    os.kill(pid,signal.SIGINT)
PY
exec bash './experiments/torque200/web-preview-wsl.sh' --port 8770 --mode original
