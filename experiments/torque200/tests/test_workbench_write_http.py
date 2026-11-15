"""Versioned commands enforce method/origin/body contracts, separate from reads."""
from contextlib import contextmanager, nullcontext
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_control import TrainingController, _write
import workbench_web


@contextmanager
def server(controller):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def route(self):
            if not workbench_web.dispatch(self, artifacts=controller.artifacts, mode="original", controller=controller):
                self.send_error(404)

        do_GET = do_HEAD = do_POST = do_PUT = route

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.daemon_threads = True
    thread = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield http.server_address
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)


class WriteHTTPTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.controller = TrainingController(self.base / "artifacts", self.base / "reference", self.base)
        self.controller._locked = nullcontext
        self.controller._alive = lambda _: False
        self.destination = self.base / "save-here"
        self.destination.mkdir()

    def request(self, address, path, method="GET", data=None, headers=None):
        connection = HTTPConnection(*address, timeout=5)
        try:
            payload = json.dumps(data).encode() if data is not None else None
            connection.request(method, path, body=payload, headers=headers or ({"Content-Type":"application/json"} if payload else {}))
            response = connection.getresponse()
            body = response.read()
            return response.status, json.loads(body) if body else None
        finally:
            connection.close()

    def test_dynamic_read_contracts_and_method_rejection(self):
        with server(self.controller) as address:
            status, result = self.request(address, "/api/workbench/v1/parameters?mode=leg_force_bc")
            self.assertEqual(status, 200)
            self.assertEqual(result["effective"]["num_envs"], 512)
            self.assertTrue(all(item["editable"] for item in result["fields"]))
            for target in ("parameters", "checkpoint-options", "storage", "capabilities"):
                self.assertEqual(self.request(address, "/api/workbench/v1/" + target)[0], 200)
            status, capabilities = self.request(address, "/api/workbench/v1/capabilities?mode=original")
            self.assertEqual(status, 200)
            self.assertTrue(capabilities["legacyControls"]["multipleViewports"])
            self.assertTrue(capabilities["legacyControls"]["preview"])
            self.assertEqual(self.request(address, "/api/workbench/v1/parameters", "POST", {})[0], 405)
            self.assertEqual(self.request(address, "/api/workbench/v1/storage", "PUT", {})[0], 405)
            self.assertEqual(self.request(address, "/api/workbench/v1/parameters?mode=original&mode=torque200")[0], 400)
            self.assertEqual(self.request(address, "/api/workbench/v1/checkpoint-options?limit=100000")[0], 400)
            self.assertFalse((self.controller.root / "workbench-storage.json").exists())

    def test_storage_writes_validate_origin_fields_and_registered_open_paths(self):
        path = "/api/workbench/v1/storage"
        with server(self.controller) as address:
            bad = {"Content-Type":"application/json", "Origin":"https://foreign.invalid"}
            self.assertEqual(self.request(address, path, "POST", {"path":str(self.destination)}, bad)[0], 403)
            self.assertFalse((self.controller.root / "workbench-storage.json").exists())
            self.assertEqual(self.request(address, path, "POST", {"path":str(self.destination), "command":"x"})[0], 400)
            status, result = self.request(address, path, "POST", {"path":str(self.destination)})
            self.assertEqual(status, 200)
            self.assertEqual(result["saveRoot"], str(self.destination))
            self.assertIn("wslDistro", result)
            status, result = self.request(address, path + "/resolve", "POST", {"path":str(self.destination)})
            self.assertEqual(status, 200)
            self.assertEqual(result, {"ok":True,"path":str(self.destination)})
            self.assertEqual(self.request(address, path + "/resolve", "POST", {"path":str(self.base)})[0], 400)

    def test_active_legacy_process_remains_active_and_rejects_location_changes(self):
        run = self.controller.root / "original" / "running"
        run.mkdir(parents=True)
        _write(self.controller.root / "active.json", dict(mode="original", run_dir=str(run), pid=123))
        _write(run / "status.json", dict(mode="original", phase="training", steps=27))
        self.controller._alive = lambda _: True
        with server(self.controller) as address:
            status, capabilities = self.request(address, "/api/workbench/v1/capabilities")
            self.assertEqual(status, 200)
            self.assertFalse(capabilities["legacyControls"]["pauseTraining"])
            self.assertFalse(capabilities["legacyControls"]["saveSnapshot"])
            self.assertFalse(capabilities["legacyControls"]["parameterWrites"])
            self.assertEqual(self.request(address, "/api/workbench/v1/storage", "POST", {"path":str(self.destination)})[0], 409)
            self.assertEqual(self.request(address, "/api/workbench/v1/import", "POST", {"path":None})[0], 400)
        self.assertFalse((run / "stop.request").exists())
        self.assertFalse((self.controller.root / "workbench-storage.json").exists())


if __name__ == "__main__":
    unittest.main()
