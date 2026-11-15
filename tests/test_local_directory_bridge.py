"""HTTP directory bridge checks origin and upstream validation before native UI."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tenny.local_directories import LocalDirectories
from tenny.proxy import ConsoleBridge


class LocalDirectoryBridgeTests(unittest.TestCase):
    def setUp(self):
        self.upstream_requests = []
        received = self.upstream_requests
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                received.append(("GET", self.path, None))
                self.send({"ok": True, "wslDistro": "Ubuntu"})
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(("POST", self.path, body))
                if self.path == "/api/workbench/v1/storage/resolve":
                    if body.get("path") != "/home/robot/checkpoints":
                        return self.send({"error": "outside registered storage"}, 400)
                    return self.send({"ok": True, "path": body["path"]})
                self.send({"ok": True})
            def send(self, value, status=200):
                payload = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.picker = Mock(return_value=r"\\wsl.localhost\Ubuntu\home\robot\checkpoints")
        self.opener = Mock()
        self.directories = LocalDirectories(platform="win32", distro="Ubuntu", picker=self.picker, opener=self.opener)
        self.bridge = ConsoleBridge(f"http://127.0.0.1:{self.upstream.server_port}/", local_directories=self.directories)

    def tearDown(self):
        self.bridge.close()
        self.upstream.shutdown()
        self.upstream.server_close()

    def request(self, path="/api/local/directories", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.bridge.server.server_port, timeout=3)
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        connection.request("POST" if body is not None else "GET", path,
                           body=json.dumps(body) if body is not None else None, headers=request_headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_directory_selection_stays_local_and_does_not_change_save_location(self):
        status, result = self.request(body={"action": "select", "purpose": "save"})
        self.assertEqual(status, 200)
        self.assertEqual(result["path"], "/home/robot/checkpoints")
        self.assertFalse(result["cancelled"])
        self.assertEqual(self.upstream_requests, [("GET", "/api/workbench/v1/storage", None)])
        self.opener.assert_not_called()

    def test_open_requires_positive_fixed_upstream_directory_validation(self):
        status, result = self.request(body={"action": "open", "path": "/home/robot/checkpoints"})
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.opener.assert_called_once_with(r"\\wsl.localhost\Ubuntu\home\robot\checkpoints")
        self.assertEqual(self.upstream_requests, [("GET", "/api/workbench/v1/storage", None),
                                                 ("POST", "/api/workbench/v1/storage/resolve",
                                                  {"path": "/home/robot/checkpoints"})])

    def test_rejected_directory_cannot_launch_native_opener(self):
        status, _ = self.request(body={"action": "open", "path": "/outside/registered/storage"})
        self.assertEqual(status, 400)
        self.opener.assert_not_called()

    def test_foreign_origin_or_unknown_actions_never_reach_native_ui(self):
        self.assertEqual(self.request(body={"action": "select", "purpose": "save"},
                                      headers={"Origin": "https://foreign.example"})[0], 403)
        self.assertEqual(self.request(body={"action": "execute", "path": "anything"})[0], 400)
        self.assertEqual(self.request("/api/local/directories?other=1")[0], 400)
        self.picker.assert_not_called()
        self.opener.assert_not_called()
        self.assertEqual(self.upstream_requests, [])

    def test_capability_and_cancel_response_are_explicit(self):
        status, capabilities = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(capabilities["select"], True)
        self.picker.return_value = None
        status, result = self.request(body={"action": "select", "purpose": "import"})
        self.assertEqual(status, 200)
        self.assertTrue(result["cancelled"])
        self.assertIsNone(result["path"])

    def test_new_api_routes_are_explicitly_forwarded_and_arbitrary_routes_are_not(self):
        for name in ("parameters", "checkpoint-options", "storage", "viewports"):
            self.assertEqual(self.request("/api/workbench/v1/" + name)[0], 200)
        for name in ("storage", "import", "viewports"):
            self.assertEqual(self.request("/api/workbench/v1/" + name, body={"action": "test"})[0], 200)
        for path in ("/api/local/execute", "/api/workbench/v1/anything", "/api/workbench/v1/viewports/../../private"):
            self.assertEqual(self.request(path)[0], 404)

    def test_distribution_mismatch_disables_native_operations(self):
        self.directories._distro = "OtherDistro"
        status, capabilities = self.request()
        self.assertEqual(status, 200)
        self.assertFalse(capabilities["available"])
        self.assertEqual(self.request(body={"action": "select", "purpose": "save"})[0], 400)
        self.picker.assert_not_called()
        self.opener.assert_not_called()

    def test_import_has_a_longer_bounded_upstream_timeout(self):
        connection_type = http.client.HTTPConnection
        with patch("tenny.proxy.http.client.HTTPConnection", wraps=connection_type) as connection:
            self.assertEqual(self.request("/api/workbench/v1/import", body={"path": "/models"})[0], 200)
        self.assertTrue(any(call.kwargs.get("timeout") == 120 for call in connection.call_args_list))


if __name__ == "__main__":
    unittest.main()
