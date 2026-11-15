"""Loopback-only bridge to this launcher's WSL console; never an open proxy."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import urlsplit

from .local_directories import DirectoryError, LocalDirectories

ROUTES = {'/', '/state', '/learning-data', '/learning-curves', '/stream.mjpg', '/control', '/training',
          '/legacy', '/health/workbench'}
ROUTES.update('/workbench/' + name for name in (
    'app.mjs', 'adapter.mjs', 'transport.mjs', 'layout.mjs', 'chart-model.mjs', 'chart-view.mjs', 'styles.css', 'tokens.css'))
ROUTES.update('/api/workbench/v1/' + name for name in (
    'state', 'checkpoints', 'metrics', 'capabilities', 'catalog', 'descriptor',
    'parameters', 'checkpoint-options', 'storage', 'storage/resolve', 'import', 'viewports'))
LOCAL_DIRECTORIES = '/api/local/directories'
POST_ROUTES = frozenset({'/control', '/training', '/api/workbench/v1/storage',
                         '/api/workbench/v1/storage/resolve', '/api/workbench/v1/import',
                         '/api/workbench/v1/viewports'})


class ConsoleBridge:
    def __init__(self, target, *, port=0, local_directories=None):
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError('Invalid loopback port')
        upstream = urlsplit(target)
        if upstream.scheme != 'http' or not upstream.hostname or upstream.username or upstream.password:
            raise ValueError('Expected a local HTTP console target')
        target_host, target_port = upstream.hostname, upstream.port or 80
        directories = local_directories or LocalDirectories()

        def prepare_directories():
            connection = http.client.HTTPConnection(target_host, target_port, timeout=15)
            try:
                connection.request('GET', '/api/workbench/v1/storage')
                response = connection.getresponse()
                body = response.read(65537)
                if response.status != 200 or len(body) > 65536:
                    raise DirectoryError('无法确认目标 WSL 目录，请等待后端连接。')
                value = json.loads(body)
                if not isinstance(value, dict):
                    raise DirectoryError('后端保存目录信息无效。')
                directories.bind_target_distro(value.get('wslDistro'))
            except (OSError, http.client.HTTPException, ValueError) as error:
                if isinstance(error, DirectoryError):
                    raise
                raise DirectoryError('无法确认目标 WSL 目录，请等待后端连接。') from error
            finally:
                connection.close()

        def validated_folder(path):
            if not isinstance(path, str) or not path or len(path) > 4096:
                raise DirectoryError('目录路径无效。')
            connection = http.client.HTTPConnection(target_host, target_port, timeout=15)
            try:
                connection.request('POST', '/api/workbench/v1/storage/resolve',
                                   body=json.dumps({'path': path}), headers={'Content-Type': 'application/json'})
                response = connection.getresponse()
                body = response.read(65537)
                if response.status != 200 or len(body) > 65536:
                    raise DirectoryError('目录未通过后端验证，暂时不能打开。')
                value = json.loads(body)
                if value.get('ok') is not True or not isinstance(value.get('path'), str):
                    raise DirectoryError('目录未通过后端验证，暂时不能打开。')
                return value['path']
            except (OSError, http.client.HTTPException, ValueError) as error:
                if isinstance(error, DirectoryError):
                    raise
                raise DirectoryError('无法验证目录，请等待后端连接恢复。') from error
            finally:
                connection.close()

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

            def local_json(self, value):
                payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                if self.command != 'HEAD':
                    self.wfile.write(payload)

            def forward(self):
                host = self.headers.get('Host', '')
                if host.split(':')[0] not in {'localhost', '127.0.0.1', 'tenny.localhost'}:
                    return self.error(400, 'Invalid local host')
                if self.headers.get('Origin') not in (None, 'http://' + host):
                    return self.error(403, 'Same-origin requests required')
                parsed = urlsplit(self.path)
                if parsed.scheme or parsed.netloc or parsed.path not in ROUTES | {LOCAL_DIRECTORIES}:
                    return self.error(404, 'Unknown console route')
                body = None
                if self.command == 'POST':
                    if parsed.path not in POST_ROUTES | {LOCAL_DIRECTORIES}:
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
                    if len(body) != length or self.headers.get('Transfer-Encoding'):
                        return self.error(400, 'Invalid request body')
                if parsed.path == LOCAL_DIRECTORIES:
                    if parsed.query:
                        return self.error(400, 'Directory endpoint does not accept query parameters')
                    try:
                        if self.command in {'GET', 'HEAD'}:
                            prepare_directories()
                            return self.local_json(directories.capabilities())
                        value = json.loads(body)
                        if not isinstance(value, dict) or set(value) - {'action', 'path', 'purpose'}:
                            raise DirectoryError('目录请求格式无效。')
                        if value.get('action') not in {'select', 'open'}:
                            raise DirectoryError('目录操作无效。')
                        prepare_directories()
                        if value.get('action') == 'select':
                            result = directories.select(purpose=value.get('purpose'), initial=value.get('path'))
                        elif value.get('action') == 'open':
                            result = directories.open(validated_folder(value.get('path')))
                        else:
                            raise DirectoryError('目录操作无效。')
                        return self.local_json(result)
                    except (DirectoryError, ValueError, TypeError) as error:
                        if self.command in {'GET', 'HEAD'}:
                            return self.local_json({'available': False, 'select': False, 'open': False,
                                                    'reason': str(error) if isinstance(error, DirectoryError)
                                                    else '保存目录信息无效。'})
                        return self.error(400, str(error) if isinstance(error, DirectoryError)
                                          else '目录请求必须为有效 JSON 对象。')
                timeout = 120 if parsed.path == '/api/workbench/v1/import' else 15
                connection = http.client.HTTPConnection(target_host, target_port, timeout=timeout)
                headers_sent = False
                try:
                    connection.request(self.command, self.path, body=body,
                                       headers={'Content-Type': 'application/json'} if body is not None else {})
                    response = connection.getresponse()
                    self.send_response(response.status)
                    for name, value in response.getheaders():
                        if name.lower() in {'content-type', 'content-length', 'cache-control', 'location',
                                            'content-security-policy', 'x-content-type-options', 'allow'}:
                            self.send_header(name, value)
                    self.end_headers()
                    headers_sent = True
                    while self.command != 'HEAD':
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
            do_HEAD = forward
            do_POST = forward

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}/'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', help='Fixed upstream console URL')
    parser.add_argument('--port', type=int, default=0)
    options = parser.parse_args(argv)
    bridge = ConsoleBridge(options.target, port=options.port)
    print(bridge.url, flush=True)
    try:
        bridge.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == '__main__':
    main()
