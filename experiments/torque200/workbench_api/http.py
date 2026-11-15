"""Isolated loopback HTTP adapter for workbench artifact reads."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import parse_qs, urlsplit

from .reader import WorkbenchError, WorkbenchReader
from .mode_catalog import mode_spec


PREFIX = "/api/workbench/v1/"


class _QueryError(ValueError):
    pass


def _query(target: str, allowed: set[str]) -> dict[str, str]:
    if len(target) > 4096:
        raise _QueryError("Request target is too long")
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or parts.fragment:
        raise _QueryError("A relative API request target is required")
    try:
        parsed = parse_qs(parts.query, keep_blank_values=True,
                          strict_parsing=True, max_num_fields=10)
    except ValueError as exc:
        raise _QueryError("Invalid query string") from exc
    if parsed.keys() - allowed:
        raise _QueryError("Unknown query parameter")
    if any(len(values) != 1 or not values[0] for values in parsed.values()):
        raise _QueryError("Query parameters must have one non-empty value")
    return {key: values[0] for key, values in parsed.items()}


def _integer(query: dict[str, str], name: str, default: int) -> int:
    value = query.get(name)
    if value is None:
        return default
    if not value.isascii() or not value.isdecimal() or len(value) > 9:
        raise _QueryError(f"{name} must be a non-negative decimal integer")
    return int(value)


def read_request(reader: WorkbenchReader, target: str, *, mode: str = "original") -> dict:
    """Route read queries for either the isolated server or existing console."""
    try:
        try:
            path = urlsplit(target).path
        except ValueError as exc:
            raise _QueryError("Invalid request target") from exc
        if path == PREFIX + "capabilities":
            _query(target, set())
            return reader.capabilities()
        if path == PREFIX + "state":
            query = _query(target, {"mode"})
            return reader.state(mode=query.get("mode", mode))
        if path == PREFIX + "descriptor":
            query = _query(target, {"mode"})
            return reader.descriptor(mode=query.get("mode", mode))
        if path in {PREFIX + "checkpoints", PREFIX + "metrics"}:
            allowed = {"mode", "offset", "limit", "revision"}
            if path.endswith("/metrics"):
                allowed.add("runId")
            query = _query(target, allowed)
            options = dict(mode=query.get("mode", mode), offset=_integer(query, "offset", 0),
                           limit=_integer(query, "limit", 500 if path.endswith("/metrics") else 100),
                           revision=query.get("revision"))
            return (reader.metrics(run_id=query.get("runId"), **options) if path.endswith("/metrics")
                    else reader.checkpoints(**options))
        raise WorkbenchError("not_found", 404, "Unknown API endpoint")
    except _QueryError as exc:
        raise WorkbenchError("invalid_query", 400, str(exc)) from exc


def make_server(reader: WorkbenchReader, *, port: int = 0,
                mode: str = "original") -> ThreadingHTTPServer:
    """Bind a separate IPv4 loopback port; never attach to existing handlers.

    The caller owns serve_forever/shutdown/server_close. Construction only opens
    a socket. The reader is injected so this adapter cannot create a controller.
    """
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    # Validate the configured mode before binding a port; this performs no I/O.
    mode_spec(mode)

    class Handler(BaseHTTPRequestHandler):
        server_version = "TennyWorkbench/1"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_args):
            pass

        def _send(self, value: dict, status: int = 200, *, allow: bool = False):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            if allow:
                self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            self.close_connection = True
            if self.command != "HEAD":
                self.wfile.write(body)

        def _error(self, code: str, message: str, status: int, *, allow=False):
            self._send({"schemaVersion": 1, "error": {
                "code": code, "message": message}}, status, allow=allow)

        def send_error(self, code, message=None, explain=None):
            # BaseHTTPRequestHandler parsing failures must not return HTML.
            if code == 501:
                return self._read_only()
            self._error("http_error", "Invalid HTTP request", code)

        def do_GET(self):
            try:
                self._send(read_request(reader, self.path, mode=mode))
            except WorkbenchError as exc:
                self._error(exc.code, exc.message, exc.status)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except Exception:
                self._error("internal_error", "Unable to read workbench artifacts", 500)

        do_HEAD = do_GET

        def _read_only(self):
            self._error("read_only", "This interface only supports GET and HEAD", 405, allow=True)

        do_POST = _read_only
        do_PUT = _read_only
        do_PATCH = _read_only
        do_DELETE = _read_only
        do_OPTIONS = _read_only
        do_TRACE = _read_only
        do_CONNECT = _read_only

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server
