"""把 vendor/ 放到 sys.path，使 `import legwheel` 用本仓库快照。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor"


def ensure_plant() -> Path:
    path = str(VENDOR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return VENDOR
