"""Production dispatch through a fake console; never contacts real services."""
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from experiments.torque200 import workbench_web


@contextmanager
def server(root, ui):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if not workbench_web.dispatch(self, artifacts=root, mode="leg_force_bc"):
                calls.append((self.command, self.path))
                workbench_web._send(self, {"legacy": True})

        do_HEAD = do_GET

        def do_POST(self):
            if not workbench_web.dispatch(self, artifacts=root, mode="leg_force_bc"):
                calls.append((self.command, self.path))
                workbench_web._send(self, {"legacyControl": True})

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    with patch.object(workbench_web, "UI_ROOT", ui):
        thread.start()
        try:
            yield http.server_address, calls
        finally:
            http.shutdown()
            http.server_close()
            thread.join(timeout=2)


class ProductionDispatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="workbench-production-http-")
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.artifacts, self.ui = self.temp / "artifacts", self.temp / "ui"
        self.ui.mkdir()
        (self.ui / "index.html").write_text('<title>Tenny</title><script type="module" src="/workbench/app.mjs"></script>')
        (self.ui / "app.mjs").write_text('console.log("fixture")')

    @staticmethod
    def request(address, path, method="GET"):
        connection = HTTPConnection(*address, timeout=5)
        try:
            connection.request(method, path, body=b"{}" if method not in ("GET", "HEAD") else None)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_real_page_assets_and_head_with_same_origin_policy(self):
        with server(self.artifacts, self.ui) as (address, calls):
            status, headers, body = self.request(address, "/")
            self.assertEqual(status, 200)
            self.assertIn(b"/workbench/app.mjs", body)
            self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
            self.assertEqual(headers["Cache-Control"], "no-store")
            status, headers, body = self.request(address, "/workbench/app.mjs")
            self.assertEqual(status, 200)
            self.assertIn("javascript", headers["Content-Type"])
            status, headers, body = self.request(address, "/", "HEAD")
            self.assertEqual(status, 200)
            self.assertEqual(body, b"")
            self.assertGreater(int(headers["Content-Length"]), 0)
            self.assertEqual(calls, [])
        self.assertFalse(self.artifacts.exists())

    def test_legacy_state_commands_and_stream_are_untouched(self):
        with server(self.artifacts, self.ui) as (address, calls):
            for method, path in (("GET", "/state"), ("GET", "/learning-data"),
                                 ("GET", "/stream.mjpg"), ("POST", "/training"), ("POST", "/control")):
                self.assertEqual(self.request(address, path, method)[0], 200)
            self.assertEqual(len(calls), 5)
            self.assertEqual(self.request(address, "/legacy")[0], 200)

    def test_versioned_namespace_remains_read_only_and_reports_capabilities(self):
        with server(self.artifacts, self.ui) as (address, calls):
            status, _, body = self.request(address, "/api/workbench/v1/capabilities")
            self.assertEqual(status, 200)
            value = json.loads(body)
            self.assertIn("leg_force_bc", value["modes"])
            self.assertTrue(value["legacyControls"]["resumeLatest"])
            self.assertFalse(value["legacyControls"]["pauseTraining"])
            for method in ("POST", "PUT", "PATCH", "DELETE"):
                status, headers, body = self.request(address, "/api/workbench/v1/state", method)
                self.assertEqual(status, 405)
                self.assertEqual(headers["Allow"], "GET, HEAD")
            self.assertEqual(self.request(address, "/api/workbench/v1/state?mode=unknown")[0], 400)
            self.assertEqual(self.request(address, "/api/workbench/v1/state?mode=leg_force_bc")[0], 200)
            self.assertEqual(calls, [])

    def test_static_traversal_and_source_files_are_not_served(self):
        with server(self.artifacts, self.ui) as (address, calls):
            for path in ("/workbench/../console_train.py", "/workbench/%2e%2e/secret", "/workbench/mock-engine.mjs"):
                self.assertEqual(self.request(address, path)[0], 404)
            self.assertEqual(calls, [])

    def test_top_level_deployment_package_imports_without_ml(self):
        code = "import sys,workbench_api; assert 'leg_force_bc' in workbench_api.WorkbenchReader('missing').capabilities()['modes']; assert not any(n in sys.modules for n in ('jax','mujoco','training_control'))"
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
