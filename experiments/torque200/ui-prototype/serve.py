"""Isolated, read-only static preview. No training imports, APIs, or proxies."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent
CSP = "; ".join((
    "default-src 'none'", "script-src 'self'", "style-src 'self'",
    "img-src 'self' data:", "font-src 'self'", "connect-src 'none'",
    "object-src 'none'", "base-uri 'none'", "form-action 'none'", "frame-ancestors 'none'",
))
STATIC_FILES = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    "tokens.css": "text/css; charset=utf-8",
    "app.mjs": "text/javascript; charset=utf-8",
    "mock-engine.mjs": "text/javascript; charset=utf-8",
    "layout.mjs": "text/javascript; charset=utf-8",
    "chart-model.mjs": "text/javascript; charset=utf-8",
    "chart-view.mjs": "text/javascript; charset=utf-8",
    "checkpoint-directory.mjs": "text/javascript; charset=utf-8",
}


def make_handler(root: Path = ROOT):
    root = root.resolve()
    files = {name: (root / name, content_type) for name, content_type in STATIC_FILES.items()}
    asset_root = root / "assets"
    if asset_root.is_dir() and not asset_root.is_symlink():
        for path in asset_root.iterdir():
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                files["assets/" + path.name] = (path, content_type)

    class Handler(BaseHTTPRequestHandler):
        server_version = "TennyUIPrototype"
        sys_version = ""

        def log_message(self, *_args):
            pass

        def respond(self, status: int, body: bytes = b"", content_type: str = "text/plain; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            if status == 405:
                self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc:
                return self.respond(404, b"Not found")
            name = unquote(parsed.path)
            name = "index.html" if name == "/" else name.removeprefix("/")
            item = files.get(name)
            if item is None:
                return self.respond(404, b"Not found")
            path, content_type = item
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
                return self.respond(404, b"Not found")
            try:
                body = path.read_bytes()
            except OSError:
                return self.respond(404, b"Not found")
            return self.respond(200, body, content_type)

        do_HEAD = do_GET

        def reject_mutation(self):
            self.close_connection = True
            self.respond(405, b"Read-only UI prototype. No control API is available.")

        do_POST = reject_mutation
        do_PUT = reject_mutation
        do_PATCH = reject_mutation
        do_DELETE = reject_mutation
        do_OPTIONS = reject_mutation
        do_CONNECT = reject_mutation
        do_TRACE = reject_mutation

    return Handler


def create_server(port: int = 0, root: Path = ROOT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(root))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="Local port; 0 selects a free port.")
    parser.add_argument("--no-browser", action="store_true", help="Accepted for convenience; this server never opens a browser.")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    with create_server(args.port) as server:
        print(f"UI_PREVIEW_READY http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
