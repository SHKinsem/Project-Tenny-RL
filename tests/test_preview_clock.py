"""CPU clock/render contracts; synthetic time is not a hardware FPS benchmark."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/torque200"))
from hybrid_web import Preview, Simulator
from workbench_preview import PreviewGroup


class Vector(list):
    def tobytes(self):
        return repr(self).encode()

    def tolist(self):
        return list(self)


def make_simulator(dt=.01):
    simulator = object.__new__(Simulator)
    simulator.cfg = SimpleNamespace(ctrl_dt=dt, episode_time=6.)
    simulator.initial = SimpleNamespace(obs={"state": Vector([0.])}, done=False,
        data=SimpleNamespace(qpos=Vector([0.]), qvel=Vector([0.]), ctrl=Vector([0.]), time=0.))
    simulator.data = SimpleNamespace(qpos=Vector([0.]), qvel=Vector([0.]), ctrl=Vector([0.]))
    simulator.camera = SimpleNamespace(type=0, fixedcamid=-1, trackbodyid=0, lookat=Vector([0., 0., 0.]),
        distance=1., azimuth=0., elevation=0., orthographic=False)
    simulator.renderer = Mock(width=800, height=500, scene=SimpleNamespace(flags=[0, 0, 0]))
    simulator.renderer.render.return_value = b"pixels"
    simulator.mujoco = Mock()
    simulator.mujoco.mjtRndFlag = SimpleNamespace(mjRND_SHADOW=0, mjRND_REFLECTION=1)
    simulator.mujoco.Renderer.side_effect = lambda _model, *, width, height: Mock(
        width=width, height=height, scene=SimpleNamespace(flags=[0, 0, 0]))
    simulator.env = SimpleNamespace(mj_model=object())
    simulator.np = SimpleNamespace(asarray=lambda value: value,
                                  isfinite=lambda value: SimpleNamespace(all=lambda: True))
    simulator.Image = Mock()
    simulator.Image.fromarray.return_value.save.side_effect = lambda stream, **_kwargs: stream.write(b"jpeg")

    def step(state, _action):
        return SimpleNamespace(obs=state.obs, done=False,
            data=SimpleNamespace(qpos=Vector([state.data.qpos[0] + dt]), qvel=Vector([0.]),
                                 ctrl=Vector([0.]), time=state.data.time + dt))

    simulator.step = step
    simulator.reset()
    return simulator


class PlaybackTests(unittest.TestCase):
    def test_render_budget_counts_only_visible_worlds(self):
        worlds = []

        def factory(control):
            preview = Preview("hybrid_virtual_leg_force_100", control)
            world = make_simulator()
            worlds.append(world)
            preview.actor = SimpleNamespace(run=lambda _obs: [0.])
            preview.simulator = world
            preview.state.update(suspended=False, clients=1)
            return preview

        group = PreviewGroup("hybrid_virtual_leg_force_100", Mock(), factory)
        self.addCleanup(group.close)
        for _ in range(3):
            group.control({"action": "create"})
        for connected, size in ((4, (400, 250)), (2, (640, 400)), (1, (800, 500))):
            for index, preview in enumerate(group.previews.values()):
                preview.state["clients"] = int(index < connected)
            with patch("hybrid_web.time.monotonic", return_value=0.):
                group.tick()
            for preview in list(group.previews.values())[:connected]:
                self.assertEqual((preview.simulator.renderer.width, preview.simulator.renderer.height), size)
                self.assertEqual(preview.snapshot()["render_width"], size[0])
                self.assertEqual(preview.snapshot()["render_height"], size[1])
            self.assertEqual(len(worlds), 4)
        for preview in list(group.previews.values())[1:]:
            preview.state.update(clients=1, suspended=True)
        group.tick()
        self.assertEqual(group.primary.render_size, (800, 500))

    def test_resizing_only_replaces_renderer_and_disables_expensive_effects(self):
        world = make_simulator()
        world.mujoco.mjtRndFlag = SimpleNamespace(mjRND_SHADOW=0, mjRND_REFLECTION=1)
        world.renderer.scene.flags = [1, 1, 1]

        def renderer(_model, *, width, height):
            return Mock(width=width, height=height, scene=SimpleNamespace(flags=[1, 1, 1]))

        world.mujoco.Renderer.side_effect = renderer
        state, initial, camera, actor = world.state, world.initial, world.camera, SimpleNamespace(run=lambda _obs: [0.])
        world._advance_remainder, world.terminal_until = .003, 17.
        previous = world.renderer
        world.configure_rendering(400, 250)
        previous.close.assert_called_once_with()
        self.assertIs(world.state, state)
        self.assertIs(world.initial, initial)
        self.assertIs(world.camera, camera)
        self.assertEqual(world._advance_remainder, .003)
        self.assertEqual(world.terminal_until, 17.)
        self.assertEqual(world.renderer.scene.flags, [0, 0, 1])
        world.renderer.update_scene.side_effect = lambda *_args, **_kwargs: world.renderer.scene.flags.__setitem__(slice(None), [1, 1, 1])
        _, metadata = world.frame(actor, paused=True)
        self.assertEqual((metadata["render_width"], metadata["render_height"]), (400, 250))
        self.assertEqual(world.renderer.scene.flags, [0, 0, 1])
        world.configure_rendering(400, 250)
        _, metadata = world.frame(actor, paused=True)
        self.assertFalse(metadata["rendered"])
        self.assertEqual(world.mujoco.Renderer.call_count, 1)
        world.configure_rendering(800, 500)
        _, metadata = world.frame(actor, paused=True)
        self.assertTrue(metadata["rendered"])
        self.assertIs(world.state, state)

    def test_group_scheduler_keeps_visible_worlds_on_their_wall_clock(self):
        for connected in (1, 4):
            with self.subTest(connected=connected):
                clock, sample_times, worlds = [0.], [], []

                def factory(control):
                    preview = Preview("hybrid_virtual_leg_force_100", control)
                    world = make_simulator()
                    worlds.append(world)
                    preview.actor = SimpleNamespace(run=lambda _obs: [0.])
                    preview.simulator = world
                    preview.state.update(suspended=False, clients=int(len(worlds) <= connected))
                    return preview

                group = PreviewGroup("hybrid_virtual_leg_force_100", Mock(), factory)
                for _ in range(3):
                    group.control({"action": "create"})

                class Wake:
                    def wait(self, timeout):
                        sample_times.append(clock[0])
                        clock[0] += timeout
                        group.closed = clock[0] >= 1 - 1e-9

                    def clear(self):
                        pass

                group.wake = Wake()
                with patch("workbench_preview.time.monotonic", lambda: clock[0]):
                    group.run()
                for world in worlds[:connected]:
                    self.assertAlmostEqual(world.state.data.time, sample_times[-1])
                for world in worlds[connected:]:
                    self.assertEqual(world.state.data.time, 0.)
                self.assertEqual(sum(preview.snapshot()["frames"] for preview in group.previews.values()), 20)

    def test_four_worlds_keep_wall_clock_speed_without_changing_control_dt(self):
        for dt in (.01, .005):
            worlds = [make_simulator(dt) for _ in range(4)]
            actor = SimpleNamespace(run=lambda _obs: [0.])
            for _ in range(5):  # 5 display frames/world, but one second of physics.
                for world in worlds:
                    world.frame(actor, advance_seconds=.2)
            for world in worlds:
                self.assertAlmostEqual(world.state.data.time, 1.)
                self.assertEqual(world.cfg.ctrl_dt, dt)

    def test_overload_is_capped_and_fractional_control_time_is_retained(self):
        world = make_simulator()
        actor = SimpleNamespace(run=lambda _obs: [0.])
        _, state = world.frame(actor, advance_seconds=2.)
        self.assertAlmostEqual(state["advanced_seconds"], .25)
        self.assertAlmostEqual(state["lag_seconds"], 1.75)
        world.frame(actor, advance_seconds=.006)
        world.frame(actor, advance_seconds=.006)
        self.assertAlmostEqual(world.state.data.time, .26)

    def test_paused_frame_reuses_jpeg_until_camera_dimensions_flags_or_reset_change(self):
        world = make_simulator()
        actor = SimpleNamespace(run=lambda _obs: [0.])
        first, state = world.frame(actor, paused=True, advance_seconds=10.)
        self.assertTrue(state["rendered"])
        second, state = world.frame(actor, paused=True, advance_seconds=10.)
        self.assertIs(first, second)
        self.assertFalse(state["rendered"])
        self.assertEqual(state["lag_seconds"], 0.)
        self.assertEqual(world.renderer.render.call_count, 1)
        for change in (lambda: setattr(world.camera, "azimuth", 20),
                       lambda: setattr(world.renderer, "width", 400),
                       lambda: world.renderer.scene.flags.__setitem__(2, 1), world.reset):
            change()
            _, state = world.frame(actor, paused=True)
            self.assertTrue(state["rendered"])
        self.assertEqual(world.renderer.render.call_count, 5)

    def test_preview_excludes_pause_disconnect_and_reset_from_elapsed_time(self):
        control = Mock()
        world = make_simulator()
        preview = Preview("hybrid_virtual_leg_force_100", control)
        preview.actor = SimpleNamespace(run=lambda _obs: [0.])
        preview.simulator = world
        preview.state.update(suspended=False, clients=1)
        with patch("hybrid_web.time.monotonic", return_value=0.):
            preview.tick()
        with patch("hybrid_web.time.monotonic", return_value=.2):
            preview.tick()
        self.assertAlmostEqual(world.state.data.time, .2)
        self.assertAlmostEqual(preview.state["playback_rate"], 1.)
        preview.state["paused"] = True
        with patch("hybrid_web.time.monotonic", return_value=10.):
            preview.tick()
        preview.state["paused"] = False
        with patch("hybrid_web.time.monotonic", return_value=20.):
            preview.tick()
        self.assertAlmostEqual(world.state.data.time, .2)
        preview.state["clients"] = 0
        with patch("hybrid_web.time.monotonic", return_value=30.):
            preview.tick()
        preview.state["clients"] = 1
        with patch("hybrid_web.time.monotonic", return_value=40.):
            preview.tick()
        self.assertAlmostEqual(world.state.data.time, .2)
        preview.reset_requested = True
        with patch("hybrid_web.time.monotonic", return_value=50.):
            preview.tick()
        self.assertEqual(world.state.data.time, 0.)
        self.assertEqual(preview.state["lag_seconds"], 0.)
        preview.release()


if __name__ == "__main__":
    unittest.main()
