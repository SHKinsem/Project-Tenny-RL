"""Only talks to its own temporary, isolated HTTP server."""

from __future__ import annotations

import http.client
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest


spec = importlib.util.spec_from_file_location("ui_preview_server", Path(__file__).with_name("serve.py"))
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class PreviewIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="tenny-ui-prototype-")
        cls.root = Path(cls.temp.name)
        for name in preview.STATIC_FILES:
            (cls.root / name).write_text("<p>demo</p>" if name == "index.html" else "// demo", encoding="utf-8")
        (cls.root / "secret.txt").write_text("must not be served", encoding="utf-8")
        (cls.root / "assets").mkdir()
        (cls.root / "assets" / "frame.png").write_bytes(b"demo image")
        (cls.root / "assets" / "private.json").write_text("{}", encoding="utf-8")
        cls.server = preview.create_server(root=cls.root)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.temp.cleanup()

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_allowlisted_static_files_and_mime(self):
        self.assertEqual(self.request("GET", "/")[0], 200)
        status, headers, _ = self.request("GET", "/mock-engine.mjs")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/javascript"))
        self.assertEqual(self.request("GET", "/assets/frame.png")[0], 200)

    def test_no_api_proxy_directory_or_source_access(self):
        for path in ("/state", "/learning-data", "/training", "/control", "/assets/", "/serve.py", "/secret.txt", "/assets/private.json"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)

    def test_mutations_are_rejected(self):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            for path in ("/training", "/control", "/"):
                with self.subTest(method=method, path=path):
                    status, headers, _ = self.request(method, path, b'{"action":"start"}')
                    self.assertEqual(status, 405)
                    self.assertEqual(headers["Allow"], "GET, HEAD")

    def test_traversal_and_absolute_proxy_targets_are_denied(self):
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/%2e%2e/index.html", "/%252e%252e/secret.txt", "http://example.invalid/index.html", "//example.invalid/index.html"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)

    def test_csp_blocks_connections_and_external_dependencies(self):
        for path in ("/", "/state"):
            _, headers, _ = self.request("GET", path)
            policy = headers["Content-Security-Policy"]
            self.assertIn("connect-src 'none'", policy)
            self.assertIn("script-src 'self'", policy)
            self.assertIn("form-action 'none'", policy)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(headers["Cache-Control"], "no-store")

    def test_head_returns_headers_without_body_and_bind_is_loopback(self):
        status, headers, body = self.request("HEAD", "/")
        self.assertEqual(status, 200)
        self.assertGreater(int(headers["Content-Length"]), 0)
        self.assertEqual(body, b"")
        self.assertEqual(self.server.server_address[0], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
