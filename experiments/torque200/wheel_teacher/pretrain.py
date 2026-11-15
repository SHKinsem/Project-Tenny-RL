"""Bounded teacher-data -> behavior-cloning initialization for Brax PPO.

All ML imports are lazy. Dataset, BC, and native replay evidence are independent
atomic stages. A short successful run proves plumbing, not stable balance.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

SCHEMA = "wheel-teacher-bc-preparation-v1"


class PreparationBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparationConfig:
    episodes: int = 8
    steps_per_episode: int = 600
    updates: int = 256
    batch_size: int = 256
    learning_rate: float = 1e-3
    exploration_std: float = .15
    budget_seconds: float = 180.0
    replay_episodes: int = 2
    replay_steps: int = 100
    seed: int = 0

    def __post_init__(self):
        bounds = {"episodes": (4, 64), "steps_per_episode": (1, 600), "updates": (1, 4096),
                  "batch_size": (1, 2048), "replay_episodes": (1, 8), "replay_steps": (1, 600)}
        for name, (lo, hi) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"Invalid bounded preparation setting: {name}")
        if (type(self.seed) is not int or self.seed < 0 or
                not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .01 or
                not math.isfinite(self.exploration_std) or not .01 <= self.exploration_std <= .5 or
                not math.isfinite(self.budget_seconds) or not 0 < self.budget_seconds <= 300):
            raise ValueError("Invalid seed/rate/preparation budget")


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _file_hash(path):
    from wheel_teacher.contract import file_digest
    return file_digest(path)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(path, value):
    _atomic(path, _canonical(value))


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _read(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Invalid preparation report")
    return json.loads(path.read_text(encoding="utf8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def split_episodes(episode_ids, seed=0):
    """No episode contributes to both fitting and held-out evaluation."""
    import numpy as np
    ids = np.asarray(episode_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("Dataset requires nonnegative integer episode IDs")
    unique = np.unique(ids)
    if len(unique) < 4:
        raise ValueError("At least four complete episode identities are required")
    shuffled = np.random.default_rng(seed).permutation(unique)
    count = max(1, int(math.ceil(len(unique) * .25)))
    heldout = shuffled[:count]
    test = np.flatnonzero(np.isin(ids, heldout))
    train = np.flatnonzero(~np.isin(ids, heldout))
    if not len(train) or not len(test):
        raise ValueError("Empty episode partition")
    return train, test, dict(train_episodes=sorted(set(ids[train].tolist())),
                             heldout_episodes=sorted(set(ids[test].tolist())),
                             train_samples=len(train), heldout_samples=len(test))


def validate_dataset(data, config):
    import numpy as np
    arrays = {name: np.asarray(data[name]) for name in
              ("observations", "privileged_observations", "actions", "episode_ids")}
    n = len(arrays["episode_ids"])
    if n < config.episodes or n > config.episodes * config.steps_per_episode:
        raise ValueError("Unexpected teacher dataset size")
    for name, width in (("observations", 26), ("privileged_observations", 55), ("actions", 4)):
        value = arrays[name]
        if value.shape != (n, width) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"Invalid teacher array: {name}")
        arrays[name] = value.astype(np.float32)
    # Labels are actuator measurements. Silently clipping them would hide an
    # incorrect torque scale or a teacher that the student cannot represent.
    if np.any(np.abs(arrays["actions"]) > 1.0):
        raise ValueError("Teacher action/actual torque label exceeds the student range; no clipping allowed")
    if len(np.unique(arrays["episode_ids"])) != config.episodes:
        raise ValueError("Missing teacher episodes")
    split_episodes(arrays["episode_ids"], config.seed)
    return arrays


def noise_profile(env):
    """Uniform sensor noise variance in the exact StudentEnv observation order."""
    import numpy as np
    cfg = env._config.noise_config
    level, scales = float(cfg.level), dict(cfg.scales)
    if not math.isfinite(level) or level < 0:
        raise ValueError("Invalid PPO observation noise")
    amplitude = np.zeros(26)
    for indices, name in ((slice(4, 7), "gyro"), (slice(7, 9), "pitch"), (9, "roll"),
                          (slice(10, 14), "joint_pos"), (slice(14, 18), "joint_vel"),
                          (slice(18, 20), "linvel"), (slice(24, 26), "joint_vel")):
        scale = float(scales[name])
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("Invalid sensor noise scale")
        amplitude[indices] = level * scale
    variance = amplitude ** 2 / 3.0
    command_ranges = {}
    commands = env._config.cmd_config
    for index, name in enumerate(("lin_vel_x", "ang_vel_yaw", "height_value", "roll_value")):
        lo, hi = map(float, getattr(commands, name))
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError("Invalid PPO command range")
        command_ranges[name] = [lo, hi]
        enabled = getattr(getattr(commands, "enable", None), name, True)
        variance[index] = (hi - lo) ** 2 / 12.0 if enabled else 0.0
    privileged = np.zeros(55)
    privileged[:26] = variance
    return dict(level=level, scales=scales, command_ranges=command_ranges, actor_variance_prior=variance.tolist(),
                critic_variance_prior=privileged.tolist(),
                semantics="Uniform sensor noise and configured command-range variance added to train-only statistics; mean unchanged, summed_variance and std updated together")


def _identities(args, config, ppo_cfg):
    from wheel_teacher.contract import metadata, original_path
    contract = metadata(args.original_root)
    original = original_path(args.original_root)
    # BC optimizer changes do not discard completed native teacher trajectories.
    sources = {}
    for root in (original / "mujoco_legwheel_rl_public/legwheel",
                 original / "mujoco_legwheel_rl_public/assets",
                 original / "mujoco_esp32_robot_deploy_public/host"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                sources[str(path.relative_to(original))] = _file_hash(path)
    for name in ("teacher.py", "env.py", "contract.py"):
        sources["wheel_teacher/" + name] = _file_hash(Path(__file__).with_name(name))
    packages = {name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "brax", "flax", "optax", "mujoco", "numpy")}
    data_identity = dict(schema=SCHEMA, contract={k: v for k, v in contract.items() if k != "source_fingerprint"},
                         sources=sources, packages={k: packages[k] for k in ("mujoco", "numpy")},
                         episodes=config.episodes, steps_per_episode=config.steps_per_episode, seed=config.seed)
    network = json.loads(json.dumps(dict(ppo_cfg.network_factory)))
    bc_identity = dict(schema=SCHEMA, source_sha256=_file_hash(__file__), packages=packages,
                       network=network, normalize_observations=bool(ppo_cfg.normalize_observations),
                       updates=config.updates, batch_size=config.batch_size, learning_rate=config.learning_rate,
                       exploration_std=config.exploration_std,
                       seed=config.seed, split="episode-75/25-v1", python=sys.version,
                       jax_environment={k: v for k, v in os.environ.items() if k.startswith(("JAX_", "XLA_"))
                                        and k != "JAX_COMPILATION_CACHE_DIR"})
    if not bc_identity["normalize_observations"]:
        raise ValueError("BC initializer requires the same normalized-observation PPO configuration")
    return data_identity, bc_identity


@contextmanager
def _locked(path, check):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        while True:
            check()
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.1)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _save_dataset(path, arrays):
    import io
    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    _atomic(path, stream.getvalue())


def _dataset(args, config, directory, expected, check, cancelled, monitor):
    import numpy as np
    path, receipt = directory / "dataset.npz", directory / "data-success.json"
    try:
        saved = _read(receipt)
        if saved.get("completed") is not True or saved["identity"] != expected or saved["dataset_sha256"] != _file_hash(path):
            raise ValueError("Teacher dataset changed")
        with np.load(path, allow_pickle=False) as data:
            arrays = validate_dataset(data, config)
        return arrays, saved, True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    check()
    from wheel_teacher.teacher import collect_dataset, CollectionCancelled
    def progress(event):
        check()
        monitor.publish(phase="preparing", stage="teacher_collect", preparation_message="采集原版教师实际轮扭矩",
                        teacher_progress=event)
    try:
        data = collect_dataset(args.original_root, episodes=config.episodes, steps_per_episode=config.steps_per_episode,
                               seed=config.seed, on_progress=progress, cancelled=cancelled)
    except CollectionCancelled:
        check()  # Translate an explicit user stop into the console's StopRequested.
        raise
    arrays = validate_dataset(data, config)
    _save_dataset(path, arrays)
    report = dict(identity=expected, dataset_sha256=_file_hash(path), teacher_report=data["report"],
                  samples=len(arrays["episode_ids"]), completed=True)
    _json(receipt, report)
    check()  # Completed data remains reusable if cancellation arrives here.
    return arrays, report, False


def _networks(arrays, config, ppo_cfg, noise):
    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training.agents.ppo import networks
    from brax.training.acme import running_statistics
    train, heldout, partition = split_episodes(arrays["episode_ids"], config.seed)
    observations = dict(state=jp.asarray(arrays["observations"]),
                        privileged_state=jp.asarray(arrays["privileged_observations"]))
    sizes = {key: value.shape[1] for key, value in observations.items()}
    factory = networks.make_ppo_networks(sizes, 4, preprocess_observations_fn=running_statistics.normalize,
                                         **dict(ppo_cfg.network_factory))
    state = running_statistics.init_state({key: jax.ShapeDtypeStruct((size,), np.float32) for key, size in sizes.items()})
    normalizer = running_statistics.update(state, {key: value[train] for key, value in observations.items()})
    prior = dict(state=jp.asarray(noise["actor_variance_prior"]),
                 privileged_state=jp.asarray(noise["critic_variance_prior"]))
    variance = {key: normalizer.summed_variance[key] + len(train) * prior[key] for key in observations}
    normalizer = normalizer.replace(summed_variance=variance,
        std={key: jp.sqrt(jp.maximum(variance[key] / len(train), 1e-12)) for key in observations})
    k1, k2 = jax.random.split(jax.random.PRNGKey(config.seed))
    params = (normalizer, factory.policy_network.init(k1), factory.value_network.init(k2))
    return factory, params, observations, train, heldout, partition


def parameter_digest(params):
    import jax
    import numpy as np
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise FloatingPointError("BC parameters contain nonfinite values")
        digest.update(str(path).encode() + b"\0" + str(array.dtype).encode() + b"\0")
        digest.update(str(array.shape).encode() + b"\0" + array.tobytes())
    return digest.hexdigest()


def numpy_actor(params):
    """The checked 26->64->64->8 swish/tanh actor; no simulator/GPU calls."""
    import numpy as np
    mean, std = np.asarray(params[0].mean["state"]), np.asarray(params[0].std["state"])
    layers = sorted(params[1]["params"].items())
    if mean.shape != (26,) or std.shape != (26,) or np.any(std <= 0) or len(layers) != 3:
        raise ValueError("Unsupported BC actor layout")
    weights = []
    for (_, layer), shape in zip(layers, ((26, 64), (64, 64), (64, 8)), strict=True):
        kernel, bias = np.asarray(layer["kernel"]), np.asarray(layer["bias"])
        if kernel.shape != shape or bias.shape != (shape[1],):
            raise ValueError("Unsupported BC actor layer")
        weights.append((kernel, bias))
    def policy(observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (26,) or not np.isfinite(x).all():
            raise ValueError("Invalid student observation")
        x = (x - mean) / std
        for index, (kernel, bias) in enumerate(weights):
            x = x @ kernel + bias
            if index < 2:
                x = x / (1 + np.exp(-np.clip(x, -80, 80)))
        action = np.tanh(x[:4])
        if not np.isfinite(action).all():
            raise FloatingPointError("Nonfinite student action")
        return action
    return policy


def _fit(arrays, config, ppo_cfg, check, monitor, noise):
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax
    from flax import serialization
    factory, initial, observations, train, heldout, partition = _networks(arrays, config, ppo_cfg, noise)
    normalizer, actor, critic = initial
    distribution = factory.parametric_action_distribution
    if type(distribution).__name__ != "NormalTanhDistribution":
        raise ValueError("BC exploration initializer requires the checked tanh-normal PPO distribution")
    minimum, factor = float(distribution._min_std), float(distribution._var_scale)
    raw_std = math.log(math.expm1(config.exploration_std / factor - minimum))
    # BC fits deterministic means only. Keep exploration independent of the
    # changing shared trunk, then allow PPO to optimize this head normally.
    actor = jax.tree.map(lambda x: x, actor)
    output = actor["params"][sorted(actor["params"])[-1]]
    if output["kernel"].shape[-1] != 8:
        raise ValueError("Unexpected PPO mean/std head layout")
    output["kernel"] = output["kernel"].at[:, 4:].set(0)
    output["bias"] = output["bias"].at[4:].set(raw_std)
    targets = jp.asarray(arrays["actions"])
    def predictions(policy, indices):
        obs = {key: value[indices] for key, value in observations.items()}
        logits = factory.policy_network.apply(normalizer, policy, obs)
        return factory.parametric_action_distribution.mode(logits)
    def loss(policy, indices):
        return jp.mean(jp.square(predictions(policy, indices) - targets[indices]))
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(actor)
    @jax.jit
    def update(policy, state, indices):
        value, grads = jax.value_and_grad(loss)(policy, indices)
        updates, state = optimizer.update(grads, state, policy)
        return optax.apply_updates(policy, updates), state, value
    measure = jax.jit(predictions)
    before = np.asarray(measure(actor, jp.asarray(heldout)))
    initial_train_loss = float(loss(actor, jp.asarray(train)))
    generator = np.random.default_rng(config.seed)
    final_loss = initial_train_loss
    for step in range(config.updates):
        check()
        indices = jp.asarray(generator.choice(train, size=config.batch_size, replace=True))
        actor, opt_state, value = update(actor, opt_state, indices)
        if step == 0 or (step + 1) % 16 == 0 or step + 1 == config.updates:
            final_loss = float(value)
            if not math.isfinite(final_loss):
                raise FloatingPointError("Nonfinite BC loss")
            monitor.publish(phase="preparing", stage="teacher_fit", bc_updates=step + 1,
                            bc_total_updates=config.updates, bc_train_loss=final_loss,
                            preparation_message="用教师实际轮扭矩拟合学生策略")
    params = jax.tree.map(lambda value: np.asarray(value).copy(), jax.device_get((normalizer, actor, critic)))
    after = np.asarray(measure(actor, jp.asarray(heldout)))
    held_targets = arrays["actions"][heldout]
    final_train_loss = float(loss(actor, jp.asarray(train)))
    if not math.isfinite(final_train_loss) or final_train_loss >= initial_train_loss:
        raise ValueError("BC did not improve its training objective; no initialization published")
    host = numpy_actor(params)
    host_error = max(float(np.max(np.abs(host(arrays["observations"][index]) - after[i])))
                     for i, index in enumerate(heldout[:8]))
    if host_error > 1e-5:
        raise ValueError("NumPy preview actor differs from the fitted Brax policy")
    tree_hash = parameter_digest(params)
    payload = serialization.to_bytes(params)
    recovered = serialization.from_bytes(initial, payload)
    if parameter_digest(recovered) != tree_hash:
        raise ValueError("BC parameter save/reload mismatch")
    report = dict(completed=True, parameter_tree_sha256=tree_hash, updates=config.updates,
                  initial_train_mse=initial_train_loss, final_train_mse=final_train_loss,
                  heldout_initial_mse=float(np.mean((before-held_targets)**2)),
                  heldout_final_mse=float(np.mean((after-held_targets)**2)),
                  heldout_action_mae=np.mean(np.abs(after-held_targets), axis=0).tolist(),
                  heldout_wheel_torque_nrmse=float(np.sqrt(np.mean((after[:, 2:]-held_targets[:, 2:])**2))),
                  numpy_actor_max_error=host_error, **partition,
                  initial_exploration_pre_tanh_std=config.exploration_std,
                  exploration_semantics="Zero std-head kernels with inverse-softplus bias; trainable again during PPO",
                  raw_actor_min_std=float(np.std(arrays["observations"][train], axis=0).min()),
                  raw_actor_low_variance_dimensions=np.flatnonzero(np.std(arrays["observations"][train], axis=0) < 1e-5).tolist(),
                  fitted_actor_min_std=float(np.asarray(normalizer.std["state"]).min()),
                  normalization_noise_prior=noise,
                  normalizer_scope="Training episodes only; held-out episodes never fit normalization or gradients",
                  heldout_scope="Episode split on one fixed model and a small command schedule; not cross-randomization generalization",
                  actor_initialization="Fresh student actor; all four action dimensions fitted, no copied wheel velocity head",
                  critic_initialization="Random untrained value network, fitted observation normalization only",
                  scope="Short BC initialization, not stable-balance or sim-to-real acceptance")
    return params, payload, report


def _load_params(payload, arrays, config, ppo_cfg, noise):
    from flax import serialization
    _, template, _, _, _, _ = _networks(arrays, config, ppo_cfg, noise)
    params = serialization.from_bytes(template, payload)
    parameter_digest(params)
    return params


def prepare(args, monitor, env, ppo_cfg):
    """Return a compatible Brax (normalizer, actor, critic) initialization."""
    if getattr(args, "restore", None):
        return dict(params=None, report=dict(skipped=True, reason="Resume restores its checkpoint; BC is not rerun"))
    config = getattr(args, "pretrain_config", None) or PreparationConfig(seed=int(args.seed))
    if not isinstance(config, PreparationConfig):
        raise ValueError("pretrain_config must be a PreparationConfig")
    start = time.monotonic()
    def cancelled():
        return (Path(args.run_dir) / "stop.request").exists() or time.monotonic()-start >= config.budget_seconds
    def check():
        monitor.check_stop()
        if time.monotonic()-start >= config.budget_seconds:
            raise PreparationBudgetExceeded("Teacher/BC preparation budget reached; completed stages were preserved")
    data_id, bc_id = _identities(args, config, ppo_cfg)
    noise = noise_profile(env)
    bc_id["normalization_noise_prior"] = noise
    cache = Path(getattr(args, "pretrain_cache_root", Path(args.torque_root) / "artifacts/wheel-teacher-pretrain"))
    data_key = _hash_bytes(_canonical(data_id))
    with _locked(cache / ".lock", check):
        check()
        monitor.publish(phase="preparing", stage="teacher_collect", preparation_message="检查教师数据缓存")
        arrays, data_report, data_hit = _dataset(args, config, cache / "data" / data_key, data_id, check, cancelled, monitor)
        bc_id["dataset_sha256"] = data_report["dataset_sha256"]
        bc_key = _hash_bytes(_canonical(bc_id))
        directory = cache / "bc" / bc_key
        receipt, params_file = directory / "bc-success.json", directory / "initial-params.msgpack"
        bc_hit = False
        try:
            saved = _read(receipt)
            if (saved["report"].get("completed") is not True or
                    saved["identity"] != bc_id or saved["params_sha256"] != _file_hash(params_file)):
                raise ValueError("BC initialization changed")
            params = _load_params(params_file.read_bytes(), arrays, config, ppo_cfg, noise)
            if parameter_digest(params) != saved["report"]["parameter_tree_sha256"]:
                raise ValueError("BC parameter tree identity changed")
            bc_report = saved["report"]
            bc_hit = True
        except (OSError, ValueError, KeyError, TypeError):
            check()
            params, payload, bc_report = _fit(arrays, config, ppo_cfg, check, monitor, noise)
            _atomic(params_file, payload)
            _json(receipt, dict(identity=bc_id, params_sha256=_file_hash(params_file), report=bc_report))
        check()  # BC is already committed; replay cancellation need not refit it.
        from wheel_teacher.teacher import evaluate_student, CollectionCancelled
        monitor.publish(phase="preparing", stage="teacher_verify", preparation_message="独立回放学生和教师")
        try:
            replay = evaluate_student(args.original_root, numpy_actor(params), episodes=config.replay_episodes,
                                      steps_per_episode=config.replay_steps, seed=config.seed + 10000, cancelled=cancelled)
        except CollectionCancelled:
            check()
            raise
        check()
        report = dict(schema=SCHEMA, completed=True, data_cache_hit=data_hit, bc_cache_hit=bc_hit,
                      data_key=data_key, bc_key=bc_key, dataset_path=str(cache / "data" / data_key / "dataset.npz"),
                      initial_params_path=str(params_file), initial_params_sha256=_file_hash(params_file),
                      parameter_tree_sha256=bc_report["parameter_tree_sha256"], teacher=data_report["teacher_report"],
                      bc=bc_report, replay=replay, preparation_seconds=time.monotonic()-start,
                      scope="Teacher imitation and short replay only; held-out loss is not stable balance")
        _json(Path(args.run_dir) / "pretraining-report.json", report)
        monitor.publish(stage="teacher_bc_ready", pretraining_report=str(Path(args.run_dir) / "pretraining-report.json"),
                        preparation_message="教师模仿初始化已保存，即将开始PPO", bc_cache_hit=bc_hit, teacher_data_cache_hit=data_hit)
        return dict(params=params, report=report)
