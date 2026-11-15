"""Run the isolated read-only workbench interface on IPv4 loopback."""
from __future__ import annotations

import argparse
from pathlib import Path

from .mode_catalog import MODES
from .http import make_server
from .reader import WorkbenchReader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="Explicit artifact root containing console/<mode>/<run>; never created")
    parser.add_argument("--mode", choices=MODES, default="original")
    parser.add_argument("--port", type=int, default=0,
                        help="Separate loopback port; 0 chooses an available port (default)")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    reader = WorkbenchReader(args.artifacts)
    with make_server(reader, port=args.port, mode=args.mode) as server:
        host, port = server.server_address
        print(f"Read-only workbench: http://{host}:{port}/api/workbench/v1/state", flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
