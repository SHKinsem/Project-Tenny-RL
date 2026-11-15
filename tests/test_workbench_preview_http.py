"""Local HTTP lifecycle checks with fake simulators, never ML or training."""
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SOURCE = Path(__file__).resolve().parents[1] / "experiments/torque200"
sys.path.insert(0, str(SOURCE))
from hybrid_web import Console, Preview, make_handler

MODE = "hybrid_virtual_leg_force_100"
PATH = "/api/workbench/v1/viewports"


class ViewportHTTPTests(unittest.TestCase):
    def setUp(self):
        self.controller = Mock()
        self.controller.request_preview.return_value = {
            "mode": MODE, "kind": "saved", "model": {"run_id": "saved"}, "request_id": "one"}
        self.controller.status.return_value = {"active": False, "latest": None}
        self.simulators = []

        def create(_mode):
            result = Mock()
            result.frame.return_value = (b"jpeg", {"episode_seconds": .05, "reset_delta": 0})
            self.simulators.append(result)
            return result

        self.simulator_factory = create
        preview = Preview(MODE, self.controller, lambda mode: self.simulator_factory(mode))
        self.console = Console(SimpleNamespace(mode=MODE), self.controller, preview)
        self.console.viewports.factory = lambda value: Preview(MODE, value, lambda mode: self.simulator_factory(mode))
        self.loader = patch("hybrid_web.load_saved", side_effect=lambda *_args: (object(), {}))
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.console))
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.console.viewports.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_address[1])
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.console.viewports.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    def json(self, body=None):
        request = Request(self.base + PATH,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            return json.load(response)

    def wait_for(self, predicate):
        for _ in range(100):
            state = self.json()
            if predicate(state):
                return state
            time.sleep(.01)
        self.fail("Viewport state did not reach the expected lifecycle stage")

    def test_two_real_http_streams_are_independent_and_release_after_suspend(self):
        self.assertEqual(self.json({"action": "create"})["createdId"], "2")
        self.json({"action": "open", "id": "1"})
        self.json({"action": "open", "id": "2"})
        state = self.wait_for(lambda value: all(not item["suspended"] for item in value["viewports"]))
        streams = [item["streamUrl"] + "&generation=" + str(item["generation"])
                   for item in state["viewports"]]
        first = urlopen(self.base + streams[0], timeout=2)
        second = urlopen(self.base + streams[1], timeout=2)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        for response in (first, second):
            self.assertIn("multipart/x-mixed-replace", response.headers["Content-Type"])
            self.assertTrue(response.read(7).startswith(b"--frame"))
        self.wait_for(lambda value: all(item["frames"] > 0 and not item["loading"]["active"]
                                      for item in value["viewports"]))
        self.json({"action": "pause", "id": "1"})
        state = self.json()
        self.assertTrue(state["viewports"][0]["paused"])
        self.assertFalse(state["viewports"][1]["paused"])
        self.json({"action": "suspend", "id": "1"})
        self.json({"action": "suspend", "id": "2"})
        for _ in range(100):
            if all(result.close.call_count == 1 for result in self.simulators):
                break
            time.sleep(.01)
        self.assertEqual(len(self.simulators), 2)
        self.assertTrue(all(result.close.call_count == 1 for result in self.simulators))
        self.controller.start.assert_not_called()
        self.controller.stop.assert_not_called()

    def test_cancel_while_factory_is_blocked_never_publishes_late_window(self):
        entered, unblock = threading.Event(), threading.Event()
        allocation = Mock()

        def create(_mode):
            entered.set()
            unblock.wait(timeout=2)
            return allocation

        self.simulator_factory = create
        self.json({"action": "open", "id": "1"})
        self.assertTrue(entered.wait(timeout=1))
        state = self.json()
        self.assertEqual(state["viewports"][0]["loading"]["stage"], "loading_environment")
        self.assertIsNone(state["viewports"][0]["loading"]["progress"])
        self.json({"action": "suspend", "id": "1"})
        unblock.set()
        for _ in range(100):
            if allocation.close.called:
                break
            time.sleep(.01)
        state = self.json()["viewports"][0]
        self.assertTrue(state["suspended"])
        self.assertFalse(state["loading"]["active"])
        self.assertEqual(state["frames"], 0)
        allocation.close.assert_called_once_with()

    def test_invalid_stream_id_is_an_http_error_without_allocating(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.base + "/stream.mjpg?viewport=5", timeout=2)
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.simulators, [])
        self.controller.request_preview.assert_not_called()

    def test_close_reopen_rejects_old_generation_and_does_not_count_it_as_a_viewer(self):
        self.json({"action": "open", "id": "1"})
        first_state = self.wait_for(lambda value: not value["viewports"][0]["suspended"])["viewports"][0]
        old_url = self.base + first_state["streamUrl"] + "&generation=" + str(first_state["generation"])
        old_stream = urlopen(old_url, timeout=2)
        self.addCleanup(old_stream.close)
        self.assertTrue(old_stream.read(7).startswith(b"--frame"))
        self.json({"action": "suspend", "id": "1"})
        self.json({"action": "open", "id": "1"})
        fresh_state = self.wait_for(lambda value: not value["viewports"][0]["suspended"]
                                  and value["viewports"][0]["clients"] == 0)["viewports"][0]
        self.assertGreater(fresh_state["generation"], first_state["generation"])
        # The previous response ends even when suspend/open both occur before
        # the stream worker next checks the suspended flag.
        old_stream.read()
        with self.assertRaises(HTTPError) as error:
            urlopen(old_url, timeout=2)
        self.assertEqual(error.exception.code, 409)
        self.assertEqual(self.json()["viewports"][0]["clients"], 0)
        fresh_url = self.base + fresh_state["streamUrl"] + "&generation=" + str(fresh_state["generation"])
        fresh_stream = urlopen(fresh_url, timeout=2)
        self.addCleanup(fresh_stream.close)
        self.assertTrue(fresh_stream.read(7).startswith(b"--frame"))
        self.wait_for(lambda value: value["viewports"][0]["loading"]["stage"] == "ready")


if __name__ == "__main__":
    unittest.main()
