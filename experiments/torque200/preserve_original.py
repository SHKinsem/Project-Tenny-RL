"""Read-only fingerprint of the two original WSL sources and their policy arrays."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/jaccob/projects/tenny-reference')
SKIP = {'.venv', '.git', '__pycache__', '.pio', '.codegraph'}


def snapshot():
    result = {}
    for name in ('mujoco_legwheel_rl_public', 'mujoco_esp32_robot_deploy_public'):
        base = ROOT / name
        for path in sorted(base.rglob('*')):
            relative = path.relative_to(base)
            if any(part in SKIP for part in relative.parts) or not path.is_file():
                continue
            result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    mode, filename = sys.argv[1:]
    current = snapshot()
    if mode == 'snapshot':
        Path(filename).write_text(json.dumps(current, indent=2))
        print(f'ORIGINAL_SNAPSHOT_OK files={len(current)}')
    elif mode == 'verify':
        previous = json.loads(Path(filename).read_text())
        changes = [p for p in sorted(current.keys() | previous.keys()) if current.get(p) != previous.get(p)]
        if changes:
            raise SystemExit(f'Original sources changed: {changes}')
        print(f'ORIGINAL_PRESERVED_OK files={len(current)}')
    else:
        raise SystemExit('Expected snapshot or verify')
