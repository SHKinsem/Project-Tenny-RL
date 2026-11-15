import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "experiments/torque200"
sys.path.insert(0, str(SOURCE))
from hybrid_web import Console, Preview
from training_control import ControlError
from workbench_preview import PreviewGroup, SnapshotCoordinator, dispatch_viewports, stream_preview


MODE = "hybrid_virtual_leg_force_100"


def controller():
    result = Mock()
    result.request_preview.return_value = {"mode": MODE, "kind": "saved",
                                           "model": {"run_id": "saved"}, "request_id": "one"}
    result.status.return_value = {"active": False, "latest": None}
    return result


def simulator():
    result = Mock()
    result.frame.return_value = (b"jpeg", {"episode_seconds": .05, "reset_delta": 0})
    return result


class SnapshotLeasesTests(unittest.TestCase):
    def test_concurrent_opens_do_not_overwrite_single_snapshot_slot(self):
        control = controller()
        coordinated = SnapshotCoordinator(control)
        first = coordinated.request_preview(MODE)
        second = coordinated.request_preview(MODE)
        control.request_preview.assert_called_once_with(MODE)
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertNotEqual(first["coordinator_lease"], second["coordinator_lease"])
        coordinated.cancel_preview(first)
        control.cancel_preview.assert_not_called()
        coordinated.cancel_preview(first)  # Late duplicate release is harmless.
        control.cancel_preview.assert_not_called()
        coordinated.cancel_preview(second)
        control.cancel_preview.assert_called_once_with(control.request_preview.return_value)
        coordinated.request_preview(MODE)
        self.assertEqual(control.request_preview.call_count, 2)

    def test_mixed_modes_cannot_share_pending_snapshot(self):
        coordinated = SnapshotCoordinator(controller())
        coordinated.request_preview(MODE)
        with self.assertRaises(ControlError):
            coordinated.request_preview("leg_force_bc")

    def test_concurrent_waiters_share_bounded_supervisor_poll(self):
        control = controller()
        coordinated = SnapshotCoordinator(control)
        with patch("workbench_preview.time.monotonic", side_effect=[10, 10.2, 10.6]):
            coordinated.status(MODE)
            coordinated.status(MODE)
            coordinated.status(MODE)
        self.assertEqual(control.status.call_count, 2)


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.control = controller()
        self.simulators = []

        def make_simulator(_mode):
            result = simulator()
            self.simulators.append(result)
            return result

        self.group = PreviewGroup(MODE, self.control,
                                  lambda value: Preview(MODE, value, make_simulator))
        self.addCleanup(self.group.close)

    def test_closed_slots_are_lazy_and_limit_is_enforced(self):
        self.assertEqual(len(self.group.snapshot()["viewports"]), 1)
        for expected in ("2", "3", "4"):
            self.assertEqual(self.group.control({"action": "create"})["createdId"], expected)
        with self.assertRaises(ControlError):
            self.group.control({"action": "create"})
        self.group.tick()
        self.assertEqual(self.simulators, [])
        self.control.request_preview.assert_not_called()
        for value in self.group.snapshot()["viewports"]:
            self.assertTrue(value["suspended"])
            self.assertFalse(value["loading"]["active"])
            self.assertIsNone(value["loading"]["progress"])

    def test_two_windows_load_independent_actors_from_one_pending_request(self):
        self.group.control({"action": "create"})
        self.group.control({"action": "open", "id": "1"})
        self.group.control({"action": "open", "id": "2"})
        self.control.request_preview.assert_called_once_with(MODE)
        actors = [object(), object()]
        with patch("hybrid_web.load_saved", side_effect=[(value, {}) for value in actors]):
            self.group.tick()
        self.assertIs(self.group.get("1").actor, actors[0])
        self.assertIs(self.group.get("2").actor, actors[1])
        self.assertIsNot(self.group.get("1").simulator, self.group.get("2").simulator)
        self.control.cancel_preview.assert_called_once()
        self.group.control({"action": "pause", "id": "1"})
        self.group.control({"action": "reset", "id": "2"})
        self.assertTrue(self.group.get("1").snapshot()["paused"])
        self.assertFalse(self.group.get("2").snapshot()["paused"])
        self.assertTrue(self.group.get("2").reset_requested)
        self.assertFalse(self.group.get("1").reset_requested)

    def test_cancel_one_waiter_keeps_other_waiter_and_releases_native_resources(self):
        self.group.control({"action": "create"})
        self.group.control({"action": "open", "id": "1"})
        self.group.control({"action": "open", "id": "2"})
        self.group.control({"action": "suspend", "id": "1"})
        self.control.cancel_preview.assert_not_called()
        with patch("hybrid_web.load_saved", return_value=(object(), {})):
            self.group.tick()
        self.assertTrue(self.group.get("1").snapshot()["suspended"])
        self.assertFalse(self.group.get("2").snapshot()["suspended"])
        self.control.cancel_preview.assert_called_once()
        allocated = self.group.get("2").simulator
        self.group.control({"action": "suspend", "id": "2"})
        self.group.tick()
        allocated.close.assert_called_once_with()
        self.assertIsNone(self.group.get("2").simulator)
        self.assertIsNone(self.group.get("2").actor)

    def test_remove_closes_renderer_and_does_not_leave_orphan_stream(self):
        self.group.control({"action": "create"})
        second = self.group.get("2")
        self.group.control({"action": "open", "id": "2"})
        with patch("hybrid_web.load_saved", return_value=(object(), {})):
            self.group.tick()
        allocated = second.simulator
        self.group.control({"action": "remove", "id": "2"})
        self.assertTrue(second.closed)
        with self.assertRaises(ValueError):
            self.group.get("2")
        self.group.tick()
        allocated.close.assert_called_once_with()
        self.group.control({"action": "remove", "id": "1"})
        self.assertEqual(len(self.group.snapshot()["viewports"]), 1)
        self.assertTrue(self.group.primary.snapshot()["suspended"])

    def test_no_model_error_does_not_leak_snapshot_lease(self):
        self.control.request_preview.return_value = {"kind": "saved", "model": None}
        with self.assertRaises(ControlError):
            self.group.control({"action": "open", "id": "1"})
        self.assertEqual(self.group.controller.leases, set())
        self.control.cancel_preview.assert_called_once()

    def test_aggregate_render_budget_remains_twenty_frames_per_second(self):
        self.group.control({"action": "create"})
        for item in self.group.previews.values():
            item.state["suspended"] = False
        observed = []

        def wait(timeout):
            observed.append(timeout)
            self.group.closed = True

        self.group.wake = Mock()
        self.group.wake.wait.side_effect = wait
        with patch.object(self.group, "tick"), patch("workbench_preview.time.monotonic", side_effect=[0, 0]):
            self.group.run()
        self.assertEqual(observed, [.1])

    def test_invalid_identifier_does_not_fall_back_to_primary(self):
        for value in ("0", "5", "../1", True, 1, ""):
            with self.assertRaises(ValueError):
                self.group.control({"action": "open", "id": value})
        self.control.request_preview.assert_not_called()


class LoadingTests(unittest.TestCase):
    def test_all_preload_phases_have_indeterminate_progress_until_first_frame(self):
        control = controller()
        allocation = simulator()
        preview = Preview(MODE, control, Mock(return_value=allocation))
        preview.open()
        self.assertEqual(preview.snapshot()["loading"],
                         {"active": True, "stage": "loading_policy", "progress": None})

        def load(*_args):
            self.assertEqual(preview.snapshot()["loading"]["stage"], "loading_policy")
            return object(), {}

        def create(_mode):
            self.assertEqual(preview.snapshot()["loading"]["stage"], "loading_environment")
            return allocation

        preview.factory = create
        with patch("hybrid_web.load_saved", side_effect=load):
            preview.tick()
        self.assertFalse(preview.snapshot()["suspended"])
        self.assertEqual(preview.snapshot()["loading"]["stage"], "first_frame")
        self.assertTrue(preview.snapshot()["loading"]["active"])
        preview.state["clients"] = 1
        preview.tick()
        self.assertEqual(preview.snapshot()["loading"],
                         {"active": False, "stage": "ready", "progress": None})
        preview.suspend()
        preview.tick()
        allocation.close.assert_called_once_with()

    def test_cancel_during_native_setup_closes_late_renderer(self):
        control = controller()
        allocation = simulator()
        preview = Preview(MODE, control)

        def create(_mode):
            preview.suspend()
            return allocation

        preview.factory = create
        preview.open()
        with patch("hybrid_web.load_saved", return_value=(object(), {})):
            preview.tick()
        self.assertTrue(preview.snapshot()["suspended"])
        self.assertEqual(preview.snapshot()["loading"]["stage"], "idle")
        allocation.close.assert_called_once_with()
        self.assertIsNone(preview.simulator)

    def test_late_loading_phase_cannot_restore_cancelled_spinner(self):
        preview = Preview(MODE, controller())
        preview.open()
        generation = preview.generation
        preview.suspend()
        preview.loading("loading_policy", generation=generation)
        self.assertEqual(preview.snapshot()["loading"]["stage"], "idle")
        self.assertFalse(preview.snapshot()["loading"]["active"])

    def test_frame_error_releases_renderer_and_surfaces_retryable_error(self):
        allocation = simulator()
        allocation.frame.side_effect = ValueError("render failed")
        preview = Preview(MODE, controller(), lambda _mode: allocation)
        preview.open()
        preview.state["clients"] = 1
        with patch("hybrid_web.load_saved", return_value=(object(), {})):
            preview.tick()
        self.assertEqual(preview.snapshot()["loading"]["stage"], "error")
        self.assertIn("render failed", preview.snapshot()["sync_error"])
        self.assertTrue(preview.snapshot()["suspended"])
        allocation.close.assert_called_once_with()

    def test_original_baseline_is_explicit_and_unknown_training_steps_stay_unknown(self):
        control = controller()
        control.request_preview.return_value = {"kind": "saved", "model": None}
        preview = Preview("original", control, lambda _mode: simulator())
        preview.open()
        with patch("hybrid_web.load_baseline", return_value=(object(),
                {"preview_kind": "baseline", "preview_training_steps": None})) as load:
            preview.tick()
        load.assert_called_once_with("original", None)
        self.assertEqual(preview.snapshot()["preview_kind"], "baseline")
        self.assertIsNone(preview.snapshot()["preview_training_steps"])
        preview.release()


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.group = PreviewGroup(MODE, controller(), lambda value: Preview(MODE, value))
        self.addCleanup(self.group.close)

    def handler(self, path, method="GET", body=None):
        encoded = json.dumps(body).encode() if body is not None else b""
        return SimpleNamespace(path=path, command=method,
                               headers={"Content-Type": "application/json", "Content-Length": str(len(encoded))},
                               rfile=io.BytesIO(encoded), send=Mock())

    def test_list_and_create_have_only_closed_slots_and_stream_routes(self):
        handler = self.handler("/api/workbench/v1/viewports")
        self.assertTrue(dispatch_viewports(handler, self.group))
        value = handler.send.call_args.args[0]
        self.assertEqual(value["limit"], 4)
        self.assertEqual(value["viewports"][0]["streamUrl"], "/stream.mjpg?viewport=1")
        handler = self.handler("/api/workbench/v1/viewports", "POST", {"action": "create"})
        dispatch_viewports(handler, self.group)
        self.assertEqual(handler.send.call_args.args[0]["createdId"], "2")
        self.assertTrue(all(item["suspended"] for item in handler.send.call_args.args[0]["viewports"]))

    def test_invalid_stream_identity_is_not_a_different_window(self):
        for query in ("viewport=9", "viewport=1&viewport=2", "other=1", "viewport=", "viewport=../1",
                      "viewport=1&generation=", "viewport=1&generation=-1", "viewport=1&generation=+1",
                      "viewport=1&generation=1.0", "viewport=1&generation=1e2", "viewport=1&generation=01",
                      "viewport=1&generation=9007199254740992", "viewport=1&generation=1&generation=2",
                      "viewport=1&generation=１"):
            handler = self.handler("/stream.mjpg?" + query)
            dispatch_viewports(handler, self.group)
            self.assertEqual(handler.send.call_args.args[1], 400)

    def test_stale_generation_is_rejected_before_headers_without_counting_client(self):
        preview = self.group.primary
        preview.generation = 3
        handler = self.handler("/stream.mjpg?viewport=1&generation=2")
        handler.send_response = Mock()
        dispatch_viewports(handler, self.group)
        self.assertEqual(handler.send.call_args.args[1], 409)
        handler.send_response.assert_not_called()
        self.assertEqual(preview.snapshot()["clients"], 0)

    def test_header_failures_always_release_client_count(self):
        preview = self.group.primary
        for error in (BrokenPipeError("closed"), RuntimeError("failed headers")):
            handler = SimpleNamespace(send_response=Mock(), send_header=Mock(),
                                      end_headers=Mock(side_effect=error))
            if isinstance(error, BrokenPipeError):
                stream_preview(handler, preview, generation=0)
            else:
                with self.assertRaises(RuntimeError):
                    stream_preview(handler, preview, generation=0)
            self.assertEqual(preview.snapshot()["clients"], 0)

    def test_generation_change_terminates_old_stream_before_new_frame(self):
        preview = self.group.primary
        preview.state["suspended"] = False
        handler = SimpleNamespace(send_response=Mock(), send_header=Mock(), end_headers=Mock(), wfile=Mock())

        def changed(*_args, **_kwargs):
            preview.generation += 1
            preview.state["frames"] += 1
            preview.jpg = b"new-generation"

        with patch.object(preview.lock, "wait_for", side_effect=changed):
            stream_preview(handler, preview, generation=0)
        handler.wfile.write.assert_not_called()
        self.assertEqual(preview.snapshot()["clients"], 0)

    def test_body_shape_and_method_rejections_do_not_allocate(self):
        for body in ([], {"action": "create", "command": "arbitrary"}):
            handler = self.handler("/api/workbench/v1/viewports", "POST", body)
            dispatch_viewports(handler, self.group)
            self.assertEqual(handler.send.call_args.args[1], 400)
        handler = self.handler("/api/workbench/v1/viewports", "DELETE")
        dispatch_viewports(handler, self.group)
        self.assertEqual(handler.send.call_args.args[1], 405)
        self.assertEqual(len(self.group.previews), 1)

    def test_training_actions_keep_preview_pause_separate(self):
        control = controller()
        console = Console(SimpleNamespace(mode=MODE), control)
        for action in ("pause", "resume", "save"):
            console.training({"action": action, "request_id": "specific"})
            getattr(control, action).assert_called_once_with(request_id="specific")
        console.control({"action": "pause"})
        self.assertTrue(console.preview.snapshot()["paused"])
        self.assertEqual(control.pause.call_count, 1)
        console.training({"action": "start", "checkpoint": "/saved/exact", "parameters": {"num_envs": 512}})
        self.assertEqual(control.start.call_args.kwargs["checkpoint"], "/saved/exact")
        self.assertEqual(control.start.call_args.kwargs["parameters"], {"num_envs": 512})


if __name__ == "__main__":
    unittest.main()
