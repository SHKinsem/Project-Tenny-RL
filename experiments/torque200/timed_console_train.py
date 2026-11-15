"""Apply runtime settings and wall timing around the original training lifecycle.

Keeping these adapters outside console_train.py preserves the control/model
fingerprints. Runtime PPO settings are applied and recorded at the Brax boundary.
"""
import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import hashlib
import importlib
import json
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
from workbench_persistence import WorkbenchPersistence
from workbench_runtime import WorkbenchRuntime, ManualSnapshotPublisher
from workbench_parameters import FIELDS as WORKBENCH_PARAMETER_FIELDS, apply_parameters, validate_parameters


TIMING_SEMANTICS = (
    "Exclusive process-wall segments; initial reset/network and first PPO/eval "
    "include JIT/device work; rollout and SGD are combined, not GPU kernel timings"
)


def training_source_paths(args):
    """Use the model roots chosen by the core runner, including before dispatch."""
    source = (args.original_root / "mujoco_legwheel_rl_public"
              if args.mode in ("original", "wheel_torque_bc") else args.torque_root / "src")
    return source, args.original_root / "mujoco_esp32_robot_deploy_public"


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
        self._workbench = WorkbenchPersistence(args.run_dir, args.mode, restore=args.restore)
        self._workbench_parameters = {}
        self._workbench_effective_parameters = {}
        self._workbench_parameter_source = None
        self._workbench_parameter_config = None
        self._workbench_parameters_applied = False
        self._workbench_parameters_actual = {}
        self._workbench_parameter_error = None
        try:
            raw_parameters = os.environ.get("TENNY_WORKBENCH_PARAMETERS", "{}")
            if len(raw_parameters.encode("utf8")) > 64 * 1024:
                raise ValueError("Training parameter request is too large")
            self._workbench_parameters = validate_parameters(json.loads(raw_parameters))
        except (ValueError, TypeError, ImportError) as error:
            self._workbench_parameter_error = error
        self._workbench_network_config = None
        self._workbench_runtime = WorkbenchRuntime(self, clock=clock)
        super().__init__(args, clock=clock)

    @property
    def elapsed(self):
        return max(0.0, super().elapsed - self._workbench_runtime.excluded_seconds())

    def runtime_metadata(self):
        profile = self.runtime_profile.as_dict() if self.runtime_profile is not None else dict(
            requested=self._runtime_requested, effective="invalid", source="validation_failed",
            settings={}, validation_error=str(self._runtime_validation_error))
        return dict(profile,
                    effective_settings=self._runtime_settings or dict(profile["settings"]),
                    applied_to_brax=self._runtime_applied,
                    runtime_module_sha256=self._runtime_module_sha256)

    def check_stop(self):
        if self._workbench_runtime.poll_intent():
            self.publish()
        super().check_stop()
        if self._runtime_validation_error is not None:
            raise ValueError("Invalid runtime configuration: " + str(self._runtime_validation_error)) from self._runtime_validation_error
        if self._workbench_parameter_error is not None:
            raise ValueError("Invalid Workbench training parameters") from self._workbench_parameter_error

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
        self.state.update(self._workbench.state_fields())
        self.state.update(self._workbench_runtime.state_fields(),
                          workbench_parameters=dict(self._workbench_parameters),
                          workbench_effective_parameters=dict(self._workbench_effective_parameters),
                          workbench_parameters_applied=self._workbench_parameters_applied,
                          workbench_parameters_actual=dict(self._workbench_parameters_actual),
                          workbench_parameter_error="Invalid training parameter request" if self._workbench_parameter_error else None)
        settings = self._runtime_settings or (self.runtime_profile.settings if self.runtime_profile else {})
        if "num_envs" in settings:
            changes["num_envs"] = settings["num_envs"]
        if "num_eval_envs" in self._workbench_effective_parameters:
            changes["eval_envs"] = self._workbench_effective_parameters["num_eval_envs"]
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
        self._workbench_runtime.reset_budget_clock()
        super().begin_training_clock()


def applied_workbench_parameters(config, monitor):
    result = apply_parameters(config, monitor._workbench_parameters)
    monitor._workbench_effective_parameters = {field["key"]: result[field["key"]] for field in WORKBENCH_PARAMETER_FIELDS if field["key"] in result}
    return result


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
        if monitor._workbench_parameter_error is not None:
            raise ValueError("Invalid Workbench training parameters") from monitor._workbench_parameter_error
        if all(name in kwargs for name in RUNTIME_FIELDS):
            missing = set(monitor._workbench_parameters) - kwargs.keys()
            if missing:
                raise ValueError("Brax call dropped explicit training parameters: " + ", ".join(sorted(missing)))
            kwargs = apply_profile(monitor.runtime_profile, kwargs)
            if monitor._workbench_parameters and monitor._workbench_parameter_config is None:
                raise ValueError("Explicit training parameters have no saved PPO configuration")
            for name in monitor._workbench_parameters:
                source = monitor._workbench_parameter_source[name]
                expected = monitor._workbench_parameter_config[name]
                if kwargs[name] != source and kwargs[name] != expected:
                    raise ValueError("Training layer changed explicit parameter outside the saved configuration: " + name)
            kwargs = applied_workbench_parameters(kwargs, monitor)
            for name in monitor._workbench_parameters:
                if kwargs[name] != monitor._workbench_parameter_config[name]:
                    raise ValueError("Actual Brax parameter differs from saved PPO configuration: " + name)
            actual = {name: kwargs[name] for name in RUNTIME_FIELDS}
            if monitor._runtime_settings is not None and actual != monitor._runtime_settings:
                raise ValueError("Actual Brax runtime settings differ from the saved PPO configuration")
            monitor._runtime_settings = actual
            monitor._runtime_applied = True
            monitor._workbench_parameters_actual = {field["key"]: kwargs[field["key"]] for field in WORKBENCH_PARAMETER_FIELDS if field["key"] in kwargs}
            monitor._workbench_parameters_applied = True
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
                monitor._workbench_runtime.boundary(int(steps), factory, params)
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
                if set(monitor._workbench_parameters) - value.keys():
                    raise ValueError("PPO configuration is missing explicit editable parameter fields")
                parameter_source = {key: value[key] for key in monitor._workbench_parameters}
                value = applied_workbench_parameters(value, monitor)
                monitor._runtime_settings = {name: value[name] for name in RUNTIME_FIELDS}
                function(monitor.args.run_dir / "runtime_profile.json", monitor.runtime_metadata())
                monitor.publish(num_envs=value["num_envs"])
            if name == "report.json":
                value = dict(value, timing_summary=monitor.timing.snapshot(),
                             timing_summary_scope="Snapshot before report write; final sidecar is authoritative",
                             runtime_profile=monitor.runtime_metadata())
                if "num_eval_envs" in monitor._workbench_effective_parameters:
                    value["num_eval_envs"] = monitor._workbench_effective_parameters["num_eval_envs"]
                value = monitor._workbench.publish_report(value)
            elif name == "status.json":
                value = monitor._workbench.bind_record(value)
            result = function(path, value)
            if name == "ppo_config.json":
                monitor._workbench_parameter_source = parameter_source
                monitor._workbench_parameter_config = {key: value[key] for key in monitor._workbench_parameters}
            return result
    return measured


def workbench_checkpoint_save(function, monitor):
    """Record only successful periodic captures; final cutoff comes from report."""
    @wraps(function)
    def saved(*args, **kwargs):
        directory = args[0] if args else kwargs.get("path")
        steps = args[1] if len(args) > 1 else kwargs.get("step")
        elapsed = monitor.elapsed
        finalizing = monitor.state.get("phase") in ("saving", "exporting", "verifying")
        training_started = getattr(monitor, "training_clock_started", monitor.state.get("training_clock_started", True))
        result = function(*args, **kwargs)
        if not finalizing and training_started:
            if directory is not None and type(steps) is int:
                checkpoint = Path(directory) / f"{steps:012d}"
                monitor._workbench.publish_checkpoint(checkpoint, elapsed_seconds=elapsed, steps=steps)
            else:
                monitor._workbench._warn("checkpoint_capture_unavailable", "Checkpoint weights were kept, but the capture boundary could not be identified.")
        return result
    return saved


def remember_workbench_network_config(function, monitor):
    @wraps(function)
    def configured(*args, **kwargs):
        value = function(*args, **kwargs)
        monitor._workbench_network_config = value
        return value
    return configured


def manual_snapshot_builder(args, monitor, jax, checkpoint_module, extract, verifier):
    """Use the existing save/export/restore contracts only for an explicit request."""
    raw_save, raw_load, raw_export = checkpoint_module.save, extract.load, extract.export_params

    def build(run, *, steps, factory, params, elapsed_seconds, stage):
        import numpy as np
        from workbench_persistence import _atomic_json, _read_json
        spec = mode_spec(args.mode)
        config = monitor._workbench_network_config
        if config is None:
            raise RuntimeError("Checkpoint network configuration is not ready")
        stage("checkpoint")
        checkpoint = run / "checkpoints" / f"{steps:012d}"
        with monitor.timing.span("workbench.save_checkpoint", steps=steps):
            raw_save(str(checkpoint.parent), steps, params, config)
        recovered = raw_load(str(checkpoint))
        if args.mode == "leg_force_bc":
            from force_teacher.ppo import restore_normalizer
            recovered[0] = restore_normalizer(recovered[0], args.experiment_config)
        checks = verifier.check_parameter_trees(params, recovered)
        stage("export")
        export = run / "policy.cpp.txt"
        export_function = raw_export
        if args.mode == "leg_force_bc":
            from force_teacher.policy import export_params
            export_function = export_params
        export_function(str(checkpoint), export)
        contract = None
        if spec["teacher"] or spec["hybrid"]:
            contract = _read_json(run / "policy_metadata.json", run)
            if spec["teacher"]:
                package = "force_teacher" if args.mode == "leg_force_bc" else "wheel_teacher"
                semantic = importlib.import_module(package + ".contract")
                digest_file, digest_tree = semantic.file_digest, semantic.tree_digest
            else:
                from hybrid.core import digest_file, digest_tree
            contract["export_sha256"] = digest_file(export)
            _atomic_json(run / "policy_metadata.json", contract, run)
        if spec["hybrid"]:
            if args.mode == "leg_force_bc":
                from force_teacher.policy import ForcePolicy
                deployed = ForcePolicy(export, run / "policy_metadata.json", original_root=args.original_root)
            else:
                from hybrid.deployment import HybridPolicy
                deployed = HybridPolicy(export, run / "policy_metadata.json",
                                        expected_mode=spec["action_mode"], expected_control_hz=spec["control_hz"])
        else:
            if args.mode in ("original", "wheel_torque_bc"):
                from host.policy.rl_policy import RL_Policy
            else:
                from legwheel.deployment.numpy_policy import RL_Policy
            deployed = RL_Policy(path=export)
        stage("verifying")
        before = jax.jit(factory(params, deterministic=True))
        after = jax.jit(factory(recovered, deterministic=True))
        mean, std = np.asarray(params[0].mean["state"]), np.asarray(params[0].std["state"])
        generator = np.random.default_rng(args.seed)
        restore_errors, export_errors = [], []
        for index in range(8):
            obs = (mean + std * generator.uniform(-1, 1, size=mean.shape)).astype(np.float32)
            key = jax.random.PRNGKey(index)
            a, b = np.asarray(before({"state": obs}, key)[0]), np.asarray(after({"state": obs}, key)[0])
            if spec["hybrid"]:
                c = deployed.action(obs)
            else:
                deployed.clean_history()
                c = deployed.run(obs)
            if not all(np.isfinite(value).all() for value in (a, b, c)):
                raise FloatingPointError("Non-finite snapshot policy action")
            restore_errors.append(float(np.max(np.abs(a - b))))
            export_errors.append(float(np.max(np.abs(a - c))))
        if max(restore_errors) > 1e-6 or max(export_errors) > 1e-3:
            raise AssertionError("Snapshot/export action comparison failed")
        report = dict(mode=args.mode, checkpoint=str(checkpoint), export_path=str(export),
                      actual_timesteps=steps, training_stop_elapsed_seconds=elapsed_seconds,
                      restore_max_abs_error=max(restore_errors), export_max_abs_error=max(export_errors),
                      export_atol=1e-3, num_eval_envs=monitor.state.get("eval_envs"),
                      optimizer_resume="Network and normalization restored; optimizer reinitialized",
                      export_observations="8 seeded synthetic observations around captured mean/std", **checks)
        if contract is not None:
            report.update(checkpoint_validated=True, action_mode=spec["action_mode"], control_hz=spec["control_hz"],
                          source_fingerprint=contract["source_fingerprint"], checkpoint_name=checkpoint.name,
                          checkpoint_sha256=digest_tree(checkpoint), export_sha256=digest_file(export))
            if spec["teacher"]:
                semantic.verified_report(run, report, args.original_root)
            else:
                from hybrid.managed_training import verified_report
                verified_report(run, args.mode, report)
        return report
    return build


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
    source, deploy = training_source_paths(args)
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
        monitor._workbench_runtime.snapshot_writer = ManualSnapshotPublisher(
            monitor, manual_snapshot_builder(args, monitor, jax, checkpoint, extract, verify_artifacts))
        with ExitStack() as stack:
            def wrap(module, name, phase, *, wait=None):
                stack.enter_context(patch.object(module, name,
                    timed_call(getattr(module, name), phase, monitor, wait=wait)))

            stack.enter_context(patch.object(train, "train", timed_ppo_train(train.train, jax, monitor)))
            stack.enter_context(patch.object(jax, "jit",
                timed_reset_jit(jax.jit, jax, monitor, callsite_code)))
            stack.enter_context(patch.object(checkpoint, "network_config",
                remember_workbench_network_config(checkpoint.network_config, monitor)))
            if selected_backend(args.mode) == "native_mujoco_cpp":
                from native_console_evaluation import evaluator_class
                stack.enter_context(patch.object(acting, "Evaluator", evaluator_class(args, monitor)))
            stack.enter_context(patch.object(acting.Evaluator, "run_evaluation",
                timed_evaluation(acting.Evaluator.run_evaluation, monitor)))
            wrap(checkpoint, "save", "save.checkpoint")
            stack.enter_context(patch.object(checkpoint, "save", workbench_checkpoint_save(checkpoint.save, monitor)))
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
        # console.main imports hybrid.managed_training before run_training.
        # A release wrapper and its model/cache root can be different directories;
        # select the latter before any model package gets its first __path__.
        previous_path = list(sys.path)
        stack.callback(sys.path.__setitem__, slice(None), previous_path)
        sys.path[:0] = [str(path) for path in training_source_paths(monitor_args)]
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
                monitor._workbench_runtime.finish()
                monitor.timing.finish(outcome=monitor.state["phase"], steps=monitor.state["steps"])
                print("TIMING_SAVED " + monitor.timing.paths["timing_text_file"], flush=True)


if __name__ == "__main__":
    main()
