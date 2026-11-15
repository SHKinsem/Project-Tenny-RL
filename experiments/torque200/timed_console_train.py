"""Apply runtime settings and wall timing around the original training lifecycle.

Keeping these adapters outside console_train.py preserves the control/model
fingerprints. Runtime PPO settings are applied and recorded at the Brax boundary.
"""
import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import hashlib
import importlib
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import console_train as console
from control_modes import mode_spec
from timing_log import RunTiming
from native_console_evaluation import selected_backend
import runtime_defaults
from runtime_defaults import apply_profile, resolve_profile, RUNTIME_FIELDS, PROFILE_CHOICES


TIMING_SEMANTICS = (
    "Exclusive process-wall segments; initial reset/network and first PPO/eval "
    "include JIT/device work; rollout and SGD are combined, not GPU kernel timings"
)


def observed_steps(monitor):
    # The parameter callback arrives before the original monitor publishes its
    # new step count; nested preview/status IO must not move the trace backwards.
    return max(monitor.state["steps"], monitor._timing_steps)


class TimedRunMonitor(console.RunMonitor):
    """Observe existing stages without changing deadlines, metrics, or stop rules."""

    def __init__(self, args, clock=time.monotonic):
        spec = mode_spec(args.mode)
        requested = getattr(args, "runtime_profile_requested", os.environ.get("TENNY_TRAINING_PROFILE", "auto"))
        self._runtime_requested = requested
        self._runtime_validation_error = None
        try:
            self.runtime_profile = resolve_profile(args.mode, restore=args.restore, requested=requested)
        except (ValueError, OSError, TypeError) as error:
            # Defer until the original lifecycle's error handler is installed,
            # so corrupt resume metadata still produces failed/timing artifacts.
            self.runtime_profile = None
            self._runtime_validation_error = error
        self._runtime_settings = None
        self._runtime_applied = False
        self._runtime_module_sha256 = hashlib.sha256(Path(runtime_defaults.__file__).read_bytes()).hexdigest()
        self.timing = RunTiming(args.run_dir, run_id=args.run_dir.name, mode=args.mode,
                                clock=clock, phase="preparation.runtime_lock"
                                if spec["teacher"] or spec["hybrid"] else "lifecycle.entry",
                                cancelled_exceptions=(console.StopRequested,))
        self._timing_in_ppo = False
        self._timing_steps = 0
        self._timing_stage = None
        self._last_timing_summary = clock()
        super().__init__(args, clock=clock)

    def runtime_metadata(self):
        profile = self.runtime_profile.as_dict() if self.runtime_profile is not None else dict(
            requested=self._runtime_requested, effective="invalid", source="validation_failed",
            settings={}, validation_error=str(self._runtime_validation_error))
        return dict(profile,
                    effective_settings=self._runtime_settings or dict(profile["settings"]),
                    applied_to_brax=self._runtime_applied,
                    runtime_module_sha256=self._runtime_module_sha256)

    def check_stop(self):
        super().check_stop()
        if self._runtime_validation_error is not None:
            raise ValueError("Invalid runtime configuration: " + str(self._runtime_validation_error)) from self._runtime_validation_error

    def publish(self, **changes):
        stage = changes.get("stage")
        if stage and stage != self._timing_stage and not self._timing_in_ppo:
            phases = {
                "imports": "setup.imports",
                "environment_setup": "setup.model_and_environment",
                "observation_shape": "setup.reset_and_networks",
                "loading_checkpoint": "preparation.restore",
                "compiling": "ppo.initialize",
                "checkpoint_and_export": "save.finalize",
                "verified": "lifecycle.completed",
                "failed": "lifecycle.failed",
                "no_model_initialized": "lifecycle.cancelled",
            }
            phase = phases.get(stage, "preparation." + stage)
            self.timing.set_phase(phase, steps=observed_steps(self), stage=stage)
            self._timing_stage = stage
        self.state.update(self.timing.paths, timing_semantics=TIMING_SEMANTICS,
                          evaluation_backend=selected_backend(self.args.mode),
                          runtime_profile=self.runtime_metadata())
        settings = self._runtime_settings or (self.runtime_profile.settings if self.runtime_profile else {})
        if "num_envs" in settings:
            changes["num_envs"] = settings["num_envs"]
        super().publish(**changes)

    def event(self, steps, metrics):
        self._timing_steps = max(self._timing_steps, int(steps))
        with self.timing.span("io.metrics", steps=int(steps)):
            super().event(steps, metrics)
        if self.clock() - self._last_timing_summary >= 10:
            self.timing.flush_summary()
            self._last_timing_summary = self.clock()

    def begin_training_clock(self):
        self.timing.set_phase("ppo.setup", steps=observed_steps(self),
                              budget_clock_reset=True, preparation_seconds=self.elapsed)
        super().begin_training_clock()


def timed_call(function, phase, monitor, *, wait=None):
    """Keep callable signatures intact for Brax's existing option introspection."""
    @wraps(function)
    def measured(*args, **kwargs):
        with monitor.timing.span(phase, steps=observed_steps(monitor),
                                 stage=monitor.state["stage"]):
            result = function(*args, **kwargs)
            return wait(result) if wait else result
    return measured


def timed_ppo_train(function, jax, monitor):
    """Observe the safe parameter callback after its GPU computation completes."""
    @wraps(function)
    def measured(*args, **kwargs):
        if monitor._runtime_validation_error is not None:
            raise ValueError("Invalid runtime configuration") from monitor._runtime_validation_error
        if all(name in kwargs for name in RUNTIME_FIELDS):
            kwargs = apply_profile(monitor.runtime_profile, kwargs)
            actual = {name: kwargs[name] for name in RUNTIME_FIELDS}
            if monitor._runtime_settings is not None and actual != monitor._runtime_settings:
                raise ValueError("Actual Brax runtime settings differ from the saved PPO configuration")
            monitor._runtime_settings = actual
            monitor._runtime_applied = True
            monitor.publish(num_envs=actual["num_envs"])
            console.atomic_json(monitor.args.run_dir / "runtime_profile.json", monitor.runtime_metadata())
        elif monitor.runtime_profile.settings or monitor._runtime_settings is not None:
            raise ValueError("Brax call is missing required runtime configuration fields")
        capture = kwargs.get("policy_params_fn")
        progress = kwargs.get("progress_fn")

        if capture is not None:
            @wraps(capture)
            def completed(steps, factory, params):
                # device_get in the original callback must wait anyway. Charge
                # that wait to PPO, before measuring the host snapshot/checks.
                jax.block_until_ready(params)
                monitor._timing_steps = max(monitor._timing_steps, int(steps))
                with monitor.timing.span("callback.snapshot_and_checks", steps=int(steps),
                                         synchronization="params ready before CPU snapshot"):
                    result = capture(steps, factory, params)
                monitor.timing.set_phase("ppo.first_epoch" if int(steps) == 0 else "ppo.train",
                                         steps=int(steps), compilation_included=int(steps) == 0,
                                         work="rollout and PPO updates together")
                return result
            kwargs["policy_params_fn"] = completed

        if progress is not None:
            kwargs["progress_fn"] = timed_call(progress, "callback.progress", monitor)

        monitor._timing_in_ppo = True
        try:
            with monitor.timing.span("ppo.initialize", steps=observed_steps(monitor),
                                     compilation_included=True):
                return function(*args, **kwargs)
        finally:
            monitor._timing_in_ppo = False
            monitor.timing.set_phase("lifecycle.training_return", steps=observed_steps(monitor))
    return measured


def timed_evaluation(function, monitor):
    evaluations = 0

    @wraps(function)
    def measured(*args, **kwargs):
        nonlocal evaluations
        first = evaluations == 0
        with monitor.timing.span("ppo.evaluate_first" if first else "ppo.evaluate",
                                 steps=observed_steps(monitor), compilation_included=first,
                                 synchronization="Brax evaluator waits for completed rollout"):
            result = function(*args, **kwargs)
        evaluations += 1
        return result
    return measured


def timed_atomic_json(function, monitor):
    @wraps(function)
    def measured(path, value):
        name = Path(path).name
        phase = {"status.json": "io.status", "progress.json": "io.progress",
                 "report.json": "io.report"}.get(name, "io.config")
        with monitor.timing.span(phase, steps=observed_steps(monitor), file=name):
            if name == "ppo_config.json":
                value = apply_profile(monitor.runtime_profile, value)
                monitor._runtime_settings = {name: value[name] for name in RUNTIME_FIELDS}
                function(monitor.args.run_dir / "runtime_profile.json", monitor.runtime_metadata())
                monitor.publish(num_envs=value["num_envs"])
            if name == "report.json":
                value = dict(value, timing_summary=monitor.timing.snapshot(),
                             timing_summary_scope="Snapshot before report write; final sidecar is authoritative",
                             runtime_profile=monitor.runtime_metadata())
            return function(path, value)
    return measured


def timed_reset_jit(function, jax, monitor, callsite_code):
    """Wait only for the runner's one setup reset, never each training reset."""
    @wraps(function)
    def measured(fun=None, *args, **kwargs):
        compiled = function(fun, *args, **kwargs)
        if (sys._getframe(1).f_code is not callsite_code or
                getattr(fun, "__name__", None) != "reset"):
            return compiled

        @wraps(compiled)
        def completed(*call_args, **call_kwargs):
            with monitor.timing.span("setup.reset_first", steps=0, compilation_included=True,
                                     synchronization="single setup reset waits for completion"):
                return jax.block_until_ready(compiled(*call_args, **call_kwargs))
        return completed
    return measured


@contextmanager
def training_hooks(args, monitor, callsite_code):
    """Wrap public boundaries before the original runner binds its imports."""
    spec = mode_spec(args.mode)
    source = (args.original_root / "mujoco_legwheel_rl_public"
              if args.mode in ("original", "wheel_torque_bc") else args.torque_root / "src")
    deploy = args.original_root / "mujoco_esp32_robot_deploy_public"
    previous_path = list(sys.path)
    sys.path[:0] = [str(source), str(deploy)]
    monitor.timing.set_phase("setup.imports", steps=0)
    try:
        import jax
        from brax.training import acting
        from brax.training.agents.ppo import checkpoint, networks, train
        import legwheel
        from legwheel import utils
        from legwheel.deployment import extract
        import preview_sync
        import verify_artifacts

        if not Path(legwheel.__file__).resolve().is_relative_to(source.resolve()):
            raise RuntimeError("Timing hooks imported the wrong legwheel source")
        with ExitStack() as stack:
            def wrap(module, name, phase, *, wait=None):
                stack.enter_context(patch.object(module, name,
                    timed_call(getattr(module, name), phase, monitor, wait=wait)))

            stack.enter_context(patch.object(train, "train", timed_ppo_train(train.train, jax, monitor)))
            stack.enter_context(patch.object(jax, "jit",
                timed_reset_jit(jax.jit, jax, monitor, callsite_code)))
            if selected_backend(args.mode) == "native_mujoco_cpp":
                from native_console_evaluation import evaluator_class
                stack.enter_context(patch.object(acting, "Evaluator", evaluator_class(args, monitor)))
            stack.enter_context(patch.object(acting.Evaluator, "run_evaluation",
                timed_evaluation(acting.Evaluator.run_evaluation, monitor)))
            wrap(checkpoint, "save", "save.checkpoint")
            wrap(utils, "save_cpu_checkpoint", "save.cpu_params")
            wrap(extract, "load", "artifact.load", wait=jax.block_until_ready)
            wrap(extract, "export_params", "save.export")
            wrap(networks, "make_ppo_networks", "setup.networks")
            wrap(verify_artifacts, "check_parameter_trees", "verify.parameters")
            wrap(preview_sync.SnapshotPublisher, "service", "callback.preview")

            if spec["teacher"]:
                package = "force_teacher" if args.mode == "leg_force_bc" else "wheel_teacher"
                if not args.restore:
                    pretrain = importlib.import_module(package + ".pretrain")
                    wrap(pretrain, "prepare", "preparation.teacher")
                if args.mode == "leg_force_bc":
                    policy = importlib.import_module("force_teacher.policy")
                    force_networks = importlib.import_module("force_teacher.networks")
                    wrap(policy, "export_params", "save.export")
                    wrap(force_networks, "make_ppo_networks", "setup.networks")
            yield
    finally:
        sys.path[:] = previous_path


def main(argv=None):
    """The supervisor uses this entry; direct console_train remains available."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-profile", choices=PROFILE_CHOICES,
                        default=os.environ.get("TENNY_TRAINING_PROFILE", "auto"))
    runtime_args, remaining = parser.parse_known_args(argv)
    if runtime_args.runtime_profile not in PROFILE_CHOICES:
        parser.error("Invalid TENNY_TRAINING_PROFILE")
    monitors = []
    original_training = console.run_training

    def create_monitor(*args, **kwargs):
        monitor_args = args[0] if args else kwargs["args"]
        monitor_args.runtime_profile_requested = runtime_args.runtime_profile
        monitor = TimedRunMonitor(*args, **kwargs)
        monitors.append(monitor)
        stack.enter_context(patch.object(console, "atomic_json",
                                         timed_atomic_json(console.atomic_json, monitor)))
        return monitor

    def run_training(args, monitor):
        monitor.check_stop()
        with training_hooks(args, monitor, original_training.__code__):
            return original_training(args, monitor)

    # No ML imports before the original entry has parsed arguments and published
    # status. Even an early stop or import error gets its own final timing log.
    with ExitStack() as stack:
        stack.enter_context(patch.object(console, "RunMonitor", create_monitor))
        stack.enter_context(patch.object(console, "run_training", run_training))
        try:
            return console.main(remaining)
        finally:
            for monitor in monitors:
                monitor.timing.finish(outcome=monitor.state["phase"], steps=monitor.state["steps"])
                print("TIMING_SAVED " + monitor.timing.paths["timing_text_file"], flush=True)


if __name__ == "__main__":
    main()
