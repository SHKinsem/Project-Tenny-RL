"""主机 Web：同一套 data/ 页面接到 Operator。

真机 HTTP 在 src/web/lqrwebserver.cpp。这里只换传输，映射仍走 command.py。
sim 与 real 共用这一套；MuJoCo 窗口由 host/sim/viewer.py 在主线程打开。

运行（训练仓 venv，会弹出 GLFW）：
  unset MUJOCO_GL
  $RL/.venv/bin/python tests/sim_web_server.py
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

from controller.hold_obs import HoldObs
from controller.sim_frame import SimFrame

from .state import Operator

WHEEL_R = 0.0194
TICK_DT = 0.02


class WebSession:
    def __init__(self) -> None:
        self.op = Operator()
        self.hold = HoldObs()
        self.q_l = 0.0
        self.q_r = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.last_cmd = [0.0, 0.0, 0.0, 0.0]
        self.last_out = self.hold.step(self.last_cmd, SimFrame())
        self.lock = threading.Lock()
        self._stop = False

    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def tick(self, dt: float = TICK_DT) -> None:
        with self.lock:
            cmd = list(self.op.command(self.now_ms()))
            turn_on = self.op.turn_on_edge()
            self.hold.sinyaw_scale = self.op.get_k("sinyaw_scale", 1.0)
            self.hold.hold_error_clip_enable = self.op.get_k("hold_error_clip_enable", 1.0) > 0.5
            self.hold.hold_error_clip_ds = self.op.get_k("hold_error_clip_ds", 3.0)
            self.hold.hold_error_clip_dpsi = self.op.get_k("hold_error_clip_dpsi", 1.0)
            self.q_l += cmd[0] / WHEEL_R * dt
            self.q_r += cmd[0] / WHEEL_R * dt
            self.yaw += cmd[1] * dt
            frame = SimFrame(q_l=self.q_l, q_r=self.q_r, yaw_rad=self.yaw)
            self.last_out = self.hold.step(cmd, frame, turn_on=turn_on)
            self.last_cmd = cmd
            self.op.debug = (
                f"cmd: {cmd[0]:.4f} {cmd[1]:.4f} {cmd[2]:.4f} {cmd[3]:.4f}\n"
                f"ds_true:{self.hold.ds_true:.6f} dpsi_true:{self.hold.dpsi_true:.6f} "
                f"s0:{self.hold.position_start:.6f} yaw0:{self.hold.yaw_sin_zero:.6f}\n"
            )

    def status(self) -> dict:
        with self.lock:
            return {
                "page": self.op.page,
                "joystick": list(self.op.joystick),
                "height_bar": self.op.height_bar,
                "oscillation_bar": self.op.oscillation_bar,
                "switcher": self.op.switcher,
                "command": list(self.last_cmd),
                "hold": {
                    "ds_true": self.hold.ds_true,
                    "dpsi_true": self.hold.dpsi_true,
                    "obs_ds": self.last_out["obs_ds_l"],
                    "obs_dpsi": self.last_out["obs_dpsi"],
                    "s0": self.hold.position_start,
                    "yaw0": self.hold.yaw_sin_zero,
                },
            }

    def run_ticker(self) -> None:
        while not self._stop:
            self.tick()
            time.sleep(TICK_DT)


SESSION = WebSession()


def _form(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    n = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(n).decode("utf-8") if n else ""
    q = parse_qs(raw, keep_blank_values=True)
    q.update(parse_qs(urlparse(handler.path).query, keep_blank_values=True))
    return {k: v[-1] for k, v in q.items()}


def _send(
    handler: BaseHTTPRequestHandler, code: int, body: bytes, ctype: str, *, no_store: bool = False
) -> None:
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    if no_store:
        handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _json(handler: BaseHTTPRequestHandler, obj) -> None:
    _send(handler, 200, json.dumps(obj).encode("utf-8"), "application/json")


def _k_rows(op: Operator) -> str:
    keys = list(op.k.keys())
    html = []
    for i, key in enumerate(keys):
        disabled = " disabled" if key == "sinwheel_scale" else ""
        html.append(
            f'<div class="k-item" data-k-name="{key}"><span class="k-index">K{i + 1}</span>'
            f'<span class="k-name">{key}</span><span class="k-desc"></span><span class="k-value">'
            f'<input type="text" name="K{i + 1}" value="{op.k[key]:.5f}"{disabled}></span></div>'
        )
    return "".join(html)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/control.html"):
            SESSION.op.page = "control"
            body = (DATA / "control.html").read_bytes()
            _send(self, 200, body, "text/html; charset=utf-8", no_store=True)
            return
        if path == "/control.js":
            _send(self, 200, (DATA / "control.js").read_bytes(), "application/javascript", no_store=True)
            return
        if path == "/main.js":
            _send(self, 200, (DATA / "main.js").read_bytes(), "application/javascript")
            return
        if path == "/main.html":
            SESSION.op.page = "main"
            html = (DATA / "main.html").read_text(encoding="utf-8")
            html = html.replace(
                "<!-- K values will be inserted here by the server -->",
                _k_rows(SESSION.op),
            )
            _send(self, 200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/getControlPageData":
            _json(
                self,
                {
                    "height_bar": SESSION.op.height_bar,
                    "oscillation_bar": SESSION.op.oscillation_bar,
                    "switcher": SESSION.op.switcher,
                },
            )
            return
        if path == "/getImuData":
            _json(self, {"yaw_rad": SESSION.yaw, "pitch_rad": SESSION.pitch, "roll_rad": SESSION.roll})
            return
        if path == "/getOtherString0_":
            _json(self, {"otherString0_": SESSION.op.debug})
            SESSION.op.debug = ""
            return
        if path == "/getOtherString":
            _json(self, {"otherString": ""})
            return
        if path == "/getOtherData":
            _json(self, [])
            return
        if path == "/getMatrix":
            _json(self, {"matrix": []})
            return
        if path == "/getVectorSize":
            _json(self, {"size": 0})
            return
        if path == "/status":
            _json(self, SESSION.status())
            return
        _send(self, 404, b"not found", "text/plain")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        form = _form(self)
        if path == "/joystickMove":
            SESSION.op.set_joystick(float(form.get("x", "0")), float(form.get("y", "0")), SESSION.now_ms())
            _send(self, 200, b"Joystick move received", "text/plain")
            return
        if path == "/setHeight":
            SESSION.op.height_bar = float(form.get("value", "0"))
            _send(self, 200, b"Height updated", "text/plain")
            return
        if path == "/setOscillation":
            SESSION.op.oscillation_bar = float(form.get("value", "0"))
            _send(self, 200, b"Oscillation updated", "text/plain")
            return
        if path == "/switchToggle":
            SESSION.op.switcher = form.get("state", "") == "true"
            _send(self, 200, b"Switch toggle received", "text/plain")
            return
        if path == "/setK":
            for i, key in enumerate(SESSION.op.k.keys()):
                raw = form.get(f"K{i + 1}")
                if raw is not None:
                    SESSION.op.k[key] = float(raw)
            _json(self, {"message": "K Values Updated Successfully", "otherString": "", "otherString0_": SESSION.op.debug})
            return
        if path == "/setMatrix":
            _json(self, {"message": "Matrix Updated Successfully"})
            return
        _send(self, 404, b"not found", "text/plain")


def make_server(
    host: str = "127.0.0.1", port: int = 8765, *, ticker: bool = True
) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    if ticker:
        threading.Thread(target=SESSION.run_ticker, daemon=True).start()
    return httpd


def serve_in_thread(host: str, port: int, *, ticker: bool = False) -> ThreadingHTTPServer:
    httpd = make_server(host, port, ticker=ticker)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    httpd = make_server(args.host, args.port, ticker=True)
    print(
        f"web kinematic  http://{args.host}:{args.port}/  "
        "（无 MuJoCo 窗口；要看机器人请用 tests/sim_web_server.py）"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        SESSION._stop = True
        httpd.server_close()


if __name__ == "__main__":
    main()
