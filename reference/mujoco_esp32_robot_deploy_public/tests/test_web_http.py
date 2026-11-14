"""Web HTTP 合同：同一套页面路由接到 Operator / HoldObs。"""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))

from web import server as web_server  # noqa: E402


class WebHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        web_server.SESSION = web_server.WebSession()
        cls.httpd = web_server.make_server("127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        web_server.SESSION._stop = True
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path: str):
        with urllib.request.urlopen(self.url(path), timeout=2) as r:
            return r.status, r.read()

    def post(self, path: str, body: str):
        req = urllib.request.Request(
            self.url(path),
            data=body.encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status, r.read()

    def test_control_page_and_js(self) -> None:
        st, body = self.get("/")
        self.assertEqual(st, 200)
        self.assertIn(b"Control Page", body)
        self.assertIn(b"connectionMode", body)
        self.assertIn(b"connectionLatency", body)
        self.assertIn(b'checkbox" id="httpFallback"', body)
        st, js = self.get("/control.js")
        self.assertEqual(st, 200)
        self.assertIn(b"joystickMove", js)
        self.assertIn(b"new WebSocket", js)
        self.assertIn(b"sendControlFrame", js)
        self.assertIn(b"httpFallbackAllowed", js)
        self.assertIn(b"WS disconnected", js)
        self.assertIn(b"startLatencyProbe", js)
        self.assertIn(b"localStorage", js)
        self.assertIn(b"restoreSwitcher", js)
        self.assertIn(b"location.port !== ''", js)
        self.assertIn(b"now - lastJoystickSentAt < 20", js)

    def test_unused_k_input_is_disabled(self) -> None:
        st, body = self.get("/main.html")
        self.assertEqual(st, 200)
        self.assertIn(b'name="K5" value="1.00000" disabled', body)
        self.assertIn(b'name="K6" value="1.30000">', body)
        self.assertEqual(body.count(b'class="k-name"'), len(web_server.SESSION.op.k))
        self.assertEqual(body.count(b'class="k-desc"'), len(web_server.SESSION.op.k))
        self.assertIn(b'class="k-value"', body)
        _, js = self.get("/main.js")
        self.assertIn(b"Wheel Control", js)
        self.assertIn(b"Position / Heading Hold", js)
        self.assertIn("轮速比例增益".encode(), js)

    def test_k_mapping_and_submission_match_server(self) -> None:
        import re

        _, body = self.get("/main.html")
        items = re.findall(
            r'<div class="k-item" data-k-name="([^"]+)">.*?'
            r'<input type="text" name="(K\d+)" value="([^"]+)"',
            body.decode(),
            re.S,
        )
        expected = list(web_server.SESSION.op.k.items())
        self.assertEqual(len(items), len(expected))
        for i, ((actual_key, input_name, raw), (key, value)) in enumerate(zip(items, expected)):
            self.assertEqual(actual_key, key)
            self.assertEqual(input_name, f"K{i + 1}")
            self.assertAlmostEqual(float(raw), value, places=5)

        original = dict(web_server.SESSION.op.k)
        try:
            updates = {f"K{i + 1}": f"{i + 1.25:.5f}" for i in range(len(expected))}
            st, _ = self.post("/setK", urllib.parse.urlencode(updates))
            self.assertEqual(st, 200)
            for i, key in enumerate(web_server.SESSION.op.k):
                self.assertAlmostEqual(web_server.SESSION.op.k[key], i + 1.25, places=5)
        finally:
            web_server.SESSION.op.k.clear()
            web_server.SESSION.op.k.update(original)

    def test_setk_ignores_disabled_field_omission(self) -> None:
        original = dict(web_server.SESSION.op.k)
        try:
            st, body = self.post("/setK", "K1=100")
            self.assertEqual(st, 200)
            self.assertIn(b"K Values Updated Successfully", body)
            self.assertEqual(web_server.SESSION.op.k["sinwheel_scale"], original["sinwheel_scale"])
        finally:
            web_server.SESSION.op.k.clear()
            web_server.SESSION.op.k.update(original)

    def test_get_control_page_data(self) -> None:
        st, body = self.get("/getControlPageData")
        self.assertEqual(st, 200)
        data = json.loads(body)
        self.assertIn("height_bar", data)
        self.assertIn("switcher", data)

    def test_get_imu_data(self) -> None:
        st, body = self.get("/getImuData")
        self.assertEqual(st, 200)
        data = json.loads(body)
        self.assertIn("yaw_rad", data)
        self.assertIn("pitch_rad", data)
        self.assertIn("roll_rad", data)

    def test_forward_stick_maps_vx(self) -> None:
        self.post("/joystickMove", "x=0&y=1")
        time.sleep(0.05)
        _, body = self.get("/status")
        data = json.loads(body)
        self.assertAlmostEqual(data["command"][0], 0.2, places=4)
        self.assertAlmostEqual(data["command"][1], 0.0, places=4)

    def test_back_stick_negative_vx(self) -> None:
        self.post("/joystickMove", "x=0&y=-1")
        time.sleep(0.05)
        _, body = self.get("/status")
        data = json.loads(body)
        self.assertAlmostEqual(data["command"][0], -0.2, places=4)
        self.assertAlmostEqual(data["command"][1], 0.0, places=4)

    def test_joystick_script_keeps_pure_axis_alive(self) -> None:
        _, js = self.get("/control.js")
        self.assertIn(b"setPointerCapture", js)
        self.assertNotIn(b"mouseleave", js)
        self.assertIn(b"Math.hypot(stickX, stickY)", js)

    def test_height_bar_crouch(self) -> None:
        self.post("/setHeight", "value=100")
        time.sleep(0.05)
        _, body = self.get("/status")
        data = json.loads(body)
        self.assertAlmostEqual(data["command"][2], -0.5, places=4)

    def test_switch_on(self) -> None:
        self.post("/switchToggle", "state=true")
        _, body = self.get("/getControlPageData")
        data = json.loads(body)
        self.assertTrue(data["switcher"])

    def test_hold_obs_vx_refreshes_position_anchor(self) -> None:
        self.post("/joystickMove", "x=0&y=1")
        time.sleep(0.12)
        _, body = self.get("/status")
        data = json.loads(body)
        # 前进时 |vx| 过死区，只刷 s0，观测 Δs 应接近 0。
        self.assertAlmostEqual(data["hold"]["obs_ds"], 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
