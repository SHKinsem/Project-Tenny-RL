import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tenny.proxy import ConsoleBridge


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.received = []
        received = self.received
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"mode":"hybrid_virtual_leg_force_100"}')
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.do_GET()
        self.upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.bridge = ConsoleBridge(f'http://127.0.0.1:{self.upstream.server_port}/')

    def tearDown(self):
        self.bridge.close()
        self.upstream.shutdown()
        self.upstream.server_close()

    def request(self, path='/state', method='GET', headers=None, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.bridge.server.server_port, timeout=2)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        status, payload = response.status, response.read()
        connection.close()
        return status, payload

    def test_state_and_stop_request_reach_only_fixed_upstream(self):
        status, payload = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)['mode'], 'hybrid_virtual_leg_force_100')
        status, _ = self.request('/training', 'POST', {'Content-Type': 'application/json'}, '{"action":"stop"}')
        self.assertEqual(status, 200)
        self.assertEqual(self.received, [{'action': 'stop'}])

    def test_foreign_hosts_origins_and_proxy_targets_are_rejected(self):
        self.assertEqual(self.request(headers={'Host': 'evil.example'})[0], 400)
        self.assertEqual(self.request(headers={'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.request('http://evil.example/state', headers={'Host': '127.0.0.1'})[0], 404)
        self.assertEqual(self.request('/training', 'POST', {'Content-Type': 'text/plain'}, '{}')[0], 415)
        self.assertEqual(self.received, [])

    def test_unavailable_upstream_is_recoverable_502(self):
        self.upstream.shutdown()
        self.upstream.server_close()
        self.assertEqual(self.request()[0], 502)


if __name__ == '__main__':
    unittest.main()
