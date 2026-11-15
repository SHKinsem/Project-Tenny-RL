"""End-to-end reads on temporary artifacts and a private ephemeral port."""
from contextlib import contextmanager
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from experiments.torque200.workbench_api import WorkbenchReader
from experiments.torque200.workbench_api.http import PREFIX, make_server
from experiments.torque200.workbench_api.checkpoint_metadata import capture_checkpoint, make_run_context


@contextmanager
def serving(reader, **options):
    server = make_server(reader, **options)
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise AssertionError("Test HTTP server did not stop")


class WorkbenchHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "artifacts"

    @staticmethod
    def request(address, path, method="GET", body=None):
        connection = HTTPConnection(*address, timeout=5)
        try:
            connection.request(method, path, body=body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read()
            return response.status, dict(response.getheaders()), data
        finally:
            connection.close()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixture(self):
        base = "console/original/run-001/"
        self.write(base + "status.json", {
            "mode": "original", "phase": "training", "stage": "optimization",
            "steps": 10, "elapsed_seconds": 60, "duration_seconds": 600,
            "training_clock_started": True,
        })
        self.write(base + "env_config.json", {"ctrl_dt": 0.01})
        path = self.root / base / "metrics.jsonl"
        events = [
            {"steps": 10, "elapsed_seconds": 60, "metrics": {"training/v_loss": 1e-9}},
            {"steps": 10, "elapsed_seconds": 61, "metrics": {"training/policy_loss": -0.5}},
            {"steps": 11, "elapsed_seconds": 62, "metrics": {"eval/episode_reward": 0}},
        ]
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        return path

    def test_empty_reads_and_head_do_not_create_artifacts(self):
        with serving(WorkbenchReader(self.root)) as address:
            for endpoint in ("state", "checkpoints", "metrics", "capabilities"):
                status, headers, body = self.request(address, PREFIX + endpoint)
                self.assertEqual(status, 200, body)
                self.assertEqual(json.loads(body)["schemaVersion"], 1)
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertNotIn("Access-Control-Allow-Origin", headers)
                self.assertTrue(headers["Content-Type"].startswith("application/json"))
            status, headers, body = self.request(address, PREFIX + "state", "HEAD")
            self.assertEqual(status, 200)
            self.assertEqual(body, b"")
            self.assertGreater(int(headers["Content-Length"]), 0)
        self.assertFalse(self.root.exists())

    def test_all_control_methods_are_rejected_and_allow_is_explicit(self):
        with serving(WorkbenchReader(self.root)) as address:
            for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT", "BREW"):
                status, headers, body = self.request(address, PREFIX + "state", method,
                                                     b'{"action":"start"}')
                self.assertEqual(status, 405, (method, body))
                self.assertEqual(headers["Allow"], "GET, HEAD")
                self.assertEqual(json.loads(body)["error"]["code"], "read_only")
            for path in ("/training", "/control"):
                status, _, _ = self.request(address, path, "POST", b"{}")
                self.assertEqual(status, 405)
        self.assertFalse(self.root.exists())

    def test_strict_queries_and_unknown_routes(self):
        invalid = [
            "state?mode=original&mode=torque200", "state?mode=", "state?extra=1",
            "state?mode=unknown", "state?mode", "capabilities?mode=original",
            "metrics?offset=-1", "metrics?limit=0", "metrics?limit=2001",
            "metrics?offset=1.0", "metrics?offset=%EF%BC%91",
            "metrics?runId=..%2Fescape", "metrics?runId=%2Fabsolute",
            "checkpoints?runId=run-001",
        ]
        with serving(WorkbenchReader(self.root)) as address:
            for suffix in invalid:
                status, _, body = self.request(address, PREFIX + suffix)
                self.assertEqual(status, 400, (suffix, body))
                self.assertIn("error", json.loads(body))
            for path in ("/", "/state", PREFIX + "training", PREFIX + "state/extra"):
                status, _, body = self.request(address, path)
                self.assertEqual(status, 404, body)

    def test_http_reads_original_values_and_detects_changed_pages(self):
        metrics = self.fixture()
        before = {p.relative_to(self.root).as_posix(): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        before_dirs = {p.relative_to(self.root).as_posix()
                       for p in self.root.rglob("*") if p.is_dir()}
        with serving(WorkbenchReader(self.root)) as address:
            status, _, body = self.request(address, PREFIX + "state")
            self.assertEqual(status, 200, body)
            state = json.loads(body)
            self.assertEqual(state["run"]["reportedPhase"], "training")
            self.assertIsNone(state["run"]["active"])
            self.assertIsNone(state["progress"]["cumulativePercent"])
            status, _, body = self.request(address, PREFIX + "metrics?runId=run-001&limit=2")
            self.assertEqual(status, 200, body)
            page = json.loads(body)
            self.assertEqual([p["steps"] for p in page["items"]], [10, 10])
            self.assertEqual(page["items"][0]["values"]["value_loss"], 1e-9)
            self.assertEqual(page["items"][1]["values"]["policy_loss"], -0.5)
            self.assertIsNone(page["items"][0]["recordedAtMs"])
            target = PREFIX + "metrics?runId=run-001&offset=2&revision=" + page["revision"]
            status, _, body = self.request(address, target)
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["items"][0]["values"]["episode_reward"], 0)
            after = {p.relative_to(self.root).as_posix(): p.read_bytes()
                     for p in self.root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(before_dirs, {p.relative_to(self.root).as_posix()
                                         for p in self.root.rglob("*") if p.is_dir()})
            with metrics.open("a", encoding="utf-8") as stream:
                stream.write('{"steps":12,"metrics":{}}\n')
            status, _, body = self.request(address, target)
            self.assertEqual(status, 409, body)
            status, _, body = self.request(address, PREFIX + "metrics?runId=missing")
            self.assertEqual(status, 404, body)

    def test_server_configuration_rejects_invalid_port_or_mode(self):
        reader = WorkbenchReader(self.root)
        for port in (-1, 65536, True):
            with self.assertRaises(ValueError):
                make_server(reader, port=port)
        with self.assertRaises(ValueError):
            make_server(reader, mode="unknown")

    def test_checkpoint_history_flows_through_http_without_enabling_control(self):
        policy = "console-original-v1"
        original = make_run_context(run_id="run-000", mode="original", clock_policy_id=policy,
                                    context_id="20000000-0000-4000-8000-000000000001")
        parent = capture_checkpoint(original, elapsed_seconds=600, steps=100000,
                                    checkpoint_id="20000000-0000-4000-8000-000000000002")
        resumed = make_run_context(run_id="run-001", mode="original", clock_policy_id=policy,
                                   origin="checkpoint", parent=parent,
                                   context_id="20000000-0000-4000-8000-000000000003")
        base = "console/original/run-001/"
        self.write(base + "workbench-run.json", resumed)
        status_record = dict(mode="original", phase="training", training_clock_started=True,
                             workbench_context_id=resumed["id"], elapsed_seconds=0, steps=0,
                             duration_seconds=600)
        self.write(base + "status.json", status_record)
        checkpoint = self.root / "console/original/run-000/checkpoints/000000100000"
        checkpoint.mkdir(parents=True)
        self.write("console/original/run-000/checkpoints/000000100000.workbench.json", parent)
        with serving(WorkbenchReader(self.root)) as address:
            status, _, body = self.request(address, PREFIX + "state")
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["progress"]["cumulativePercent"], 50)
            status_record.update(elapsed_seconds=60, steps=10000)
            self.write(base + "status.json", status_record)
            status, _, body = self.request(address, PREFIX + "state")
            self.assertEqual(status, 200, body)
            self.assertAlmostEqual(json.loads(body)["progress"]["cumulativePercent"], 55, places=12)
            status, _, body = self.request(address, PREFIX + "checkpoints")
            self.assertEqual(status, 200, body)
            model = json.loads(body)["items"][0]
            self.assertEqual(model["id"], parent["id"])
            self.assertEqual(model["trainedSeconds"], 600)
            self.assertEqual(model["trainedSteps"], 100000)
            self.assertFalse(model["canResume"])
            self.assertEqual(self.request(address, "/training", "POST", b"{}")[0], 405)

    def test_cli_help_has_no_artifact_or_training_side_effect(self):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-B", "-m", "experiments.torque200.workbench_api", "--help"],
            cwd=Path(__file__).resolve().parents[3], env=environment,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--artifacts", result.stdout)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
