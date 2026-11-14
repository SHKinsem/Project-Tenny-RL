"""Loopback-only bridge to this launcher's WSL console; never an open proxy."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import urlsplit

ROUTES = {'/', '/state', '/learning-data', '/learning-curves', '/stream.mjpg', '/control', '/training'}


class ConsoleBridge:
    def __init__(self, target):
        upstream = urlsplit(target)
        if upstream.scheme != 'http' or not upstream.hostname or upstream.username or upstream.password:
            raise ValueError('Expected a local HTTP console target')
        target_host, target_port = upstream.hostname, upstream.port or 80

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def error(self, code, message):
                payload = json.dumps({'error': message}).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def forward(self):
                host = self.headers.get('Host', '')
                if host.split(':')[0] not in {'localhost', '127.0.0.1', 'tenny.localhost'}:
                    return self.error(400, 'Invalid local host')
                if self.headers.get('Origin') not in (None, 'http://' + host):
                    return self.error(403, 'Same-origin requests required')
                parsed = urlsplit(self.path)
                if parsed.scheme or parsed.netloc or parsed.path not in ROUTES:
                    return self.error(404, 'Unknown console route')
                body = None
                if self.command == 'POST':
                    if parsed.path not in {'/control', '/training'}:
                        return self.error(404, 'Unknown control route')
                    if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                        return self.error(415, 'JSON required')
                    try:
                        length = int(self.headers.get('Content-Length', '0'))
                    except ValueError:
                        return self.error(400, 'Invalid request size')
                    if not 0 < length <= 65536:
                        return self.error(400, 'Invalid request size')
                    body = self.rfile.read(length)
                connection = http.client.HTTPConnection(target_host, target_port, timeout=15)
                headers_sent = False
                try:
                    connection.request(self.command, self.path, body=body,
                                       headers={'Content-Type': 'application/json'} if body is not None else {})
                    response = connection.getresponse()
                    self.send_response(response.status)
                    for name, value in response.getheaders():
                        if name.lower() in {'content-type', 'content-length', 'cache-control', 'location'}:
                            self.send_header(name, value)
                    self.end_headers()
                    headers_sent = True
                    while True:
                        block = response.read1(65536)
                        if not block:
                            break
                        self.wfile.write(block)
                        self.wfile.flush()
                except (OSError, http.client.HTTPException):
                    if not headers_sent:
                        self.error(502, '控制台正在启动或切换模式，请稍后重试。')
                finally:
                    connection.close()

            do_GET = forward
            do_POST = forward

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}/'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
