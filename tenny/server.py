"""本地 HTTP 服务：任务 API + 单页 UI。只监听本机，只跑清单里的任务。"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl

from . import registry, runner

UI = Path(__file__).with_name("ui") / "index.html"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def public(task):
    keys = ("id", "name", "group", "needs", "params", "url")
    return {k: task[k] for k in keys if k in task}


def make_server(tasks, host="127.0.0.1", port=8800):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _guard(self):
            """DNS rebinding 防护：Host 必须是本机。"""
            if self.headers.get("Host", "").rsplit(":", 1)[0] not in LOCAL_HOSTS:
                self._send(400, {"error": "bad host"})
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            path, _, qs = self.path.partition("?")
            if path == "/":
                return self._send(200, UI.read_bytes(), "text/html")
            if path == "/api/tasks":
                return self._send(200, [public(t) for t in tasks.values()])
            if path.startswith("/api/jobs/"):
                job = runner.get(path.rsplit("/", 1)[1])
                if job is None:
                    return self._send(404, {"error": "no such job"})
                try:
                    since = int(dict(parse_qsl(qs)).get("since", 0))
                except ValueError:
                    since = 0
                return self._send(200, job.snapshot(since))
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._guard():
                return
            # 非 JSON 的 POST（如跨站表单）直接拒绝；fetch 发 JSON 会触发 CORS 预检，我们不应答
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(400, {"error": "content-type must be application/json"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self._send(400, {"error": "bad json"})
            kind, _, ident = self.path.removeprefix("/api/").partition("/")
            try:
                if kind == "run":
                    return self._send(200, runner.start(ident, body, tasks).snapshot(0))
                if kind == "stop":
                    job = runner.get(ident)
                    if job is None:
                        return self._send(404, {"error": "no such job"})
                    job.stop()
                    return self._send(200, job.snapshot(0))
            except registry.UnknownTask as e:
                return self._send(404, {"error": str(e)})
            except registry.Busy as e:
                return self._send(409, {"error": str(e)})
            except registry.TaskError as e:
                return self._send(400, {"error": str(e)})
            self._send(404, {"error": "not found"})

    return ThreadingHTTPServer((host, port), Handler)
