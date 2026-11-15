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
                self.send_header('Content-Security-Policy', "connect-src 'self'")
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

    def request(self, path='/state', method='GET', headers=None, body=None, timeout=2):
        connection = http.client.HTTPConnection('127.0.0.1', self.bridge.server.server_port, timeout=timeout)
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
        # The bridge's production upstream timeout is 15 seconds; Windows may
        # retry a refused SYN longer than this test's usual 2-second client wait.
        self.assertEqual(self.request(timeout=20)[0], 502)

    def test_workbench_routes_are_explicit_without_turning_into_open_proxy(self):
        for path in ('/workbench/app.mjs', '/workbench/styles.css', '/api/workbench/v1/state?mode=leg_force_bc',
                     '/api/workbench/v1/catalog', '/api/workbench/v1/descriptor', '/legacy', '/health/workbench'):
            self.assertEqual(self.request(path)[0], 200)
        for path in ('/workbench/../console_train.py', '/workbench/private.mjs', '/api/workbench/v1/start',
                     '/api/workbench/v1/not-a-route'):
            self.assertEqual(self.request(path)[0], 404)
        connection = http.client.HTTPConnection('127.0.0.1', self.bridge.server.server_port, timeout=2)
        try:
            connection.request('GET', '/')
            response = connection.getresponse()
            self.assertEqual(response.getheader('Content-Security-Policy'), "connect-src 'self'")
            response.read()
        finally:
            connection.close()


if __name__ == '__main__':
    unittest.main()
