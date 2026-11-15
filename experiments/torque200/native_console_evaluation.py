"""Select the native evaluation seam without changing policy-source fingerprints."""
import importlib
import hashlib
import json
import os
from pathlib import Path
import sys


def selected_backend(mode, value=None):
    requested = value if value is not None else os.environ.get("TENNY_EVAL_BACKEND", "auto")
    if requested not in ("auto", "native", "gpu"):
        raise ValueError("TENNY_EVAL_BACKEND must be auto, native or gpu")
    if requested == "gpu":
        return "mjx_gpu"
    if mode == "leg_force_bc":
        return "native_mujoco_cpp"
    if requested == "native":
        raise ValueError("Native training evaluation is currently supported for leg_force_bc only")
    return "mjx_gpu"


def evaluator_class(args, monitor):
    """Return a class so the existing runner can retain its evaluation scheduler."""
    root = Path(os.environ.get("TENNY_NATIVE_EVAL_PYTHONPATH",
                              Path.home() / "projects/tenny-native-eval/python")).expanduser().resolve()
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("native_eval")
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Imported native_eval differs from TENNY_NATIVE_EVAL_PYTHONPATH")
        from native_eval import capabilities
        from native_eval.brax import BraxEvaluator, legwheel_initial_states
        from native_eval.legwheel import task_descriptor_from_config, policy_template
        available = capabilities()
        if "legwheel" not in available["tasks"] or not available.get("episode_metrics"):
            raise RuntimeError("Rebuild native evaluator: legwheel and episode_metrics are required")
        from native_eval import _native
        identity = {"extension_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()}
        for name in ("brax.py", "brax_policy.py", "legwheel.py"):
            identity[name + "_sha256"] = hashlib.sha256((Path(module.__file__).parent / name).read_bytes()).hexdigest()
    except ImportError as exc:
        raise RuntimeError("Native evaluator unavailable; run experiments/native_eval/build-wsl.sh "
                           "and check TENNY_NATIVE_EVAL_PYTHONPATH: " + str(exc)) from exc
    finally:
        sys.path.pop(0)

    class ConsoleNativeEvaluator(BraxEvaluator):
        def __init__(self, *values, **options):
            def read(name):
                return json.loads((args.run_dir / name).read_text(encoding="utf8"))
            config, metadata, ppo = read("env_config.json"), read("policy_metadata.json"), read("ppo_config.json")
            if (not ppo.get("deterministic_eval") or not ppo.get("normalize_observations") or
                    ppo.get("normalize_until_count") != 0 or
                    ppo.get("normalize_observations_mode") != "ema" or
                    ppo.get("normalize_observations_std_eps") != 0.0 or
                    ppo.get("network_factory", {}).get("policy_obs_key") != "state"):
                raise ValueError("Native console evaluation requires the supported deterministic fixed-normalizer contract")
            descriptor = task_descriptor_from_config(config, metadata, args.torque_root / "src")
            template = policy_template(metadata, ppo)
            super().__init__(*values, **options, descriptor=descriptor, policy_template=template,
                             initial_states=legwheel_initial_states, check_stop=monitor.check_stop,
                             span=lambda phase: monitor.timing.span(phase, steps=max(
                                 monitor.state["steps"], monitor._timing_steps)),
                             report=lambda value: monitor.publish(**value))
            monitor.publish(evaluation_backend="native_mujoco_cpp", native_eval_version=available["version"],
                            native_eval_identity=identity,
                            evaluation_policy_source="live training actor and normalizer; critic stays on GPU")

    return ConsoleNativeEvaluator
