"""Bounded teacher-data -> behavior-cloning initialization for Brax PPO.

All ML imports are lazy. Dataset, BC, and native replay evidence are independent
atomic stages. A short successful run proves plumbing, not stable balance.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

SCHEMA = "force-teacher-bc-preparation-v3"


class PreparationBudgetExceeded(RuntimeError):
    pass


class PreparationDataInsufficient(ValueError):
    """Reachable teacher coverage is insufficient; do not publish fake BC success."""


class PreparationGateFailed(RuntimeError):
    """A fitted actor failed physical closed-loop validation; PPO must not start."""


@dataclass(frozen=True)
class PreparationConfig:
    episodes: int = 8
    steps_per_episode: int = 600
    updates: int = 1024
    batch_size: int = 256
    learning_rate: float = 1e-3
    exploration_std: float = .01
    budget_seconds: float = 300.0
    replay_episodes: int = 3
    replay_steps: int = 600
    dagger_rounds: int = 2
    seed: int = 0

    def __post_init__(self):
        bounds = {"episodes": (4, 64), "steps_per_episode": (1, 600), "updates": (1, 4096),
                  "batch_size": (1, 2048), "replay_episodes": (1, 8), "replay_steps": (1, 600)}
        for name, (lo, hi) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"Invalid bounded preparation setting: {name}")
        if self.replay_episodes < 3 or self.replay_steps != 600:
            raise ValueError("BC gate requires at least three full six-second episodes")
        if type(self.dagger_rounds) is not int or not 0 <= self.dagger_rounds <= 2:
            raise ValueError("DAgger is limited to two rounds")
        if (type(self.seed) is not int or self.seed < 0 or
                not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .01 or
                not math.isfinite(self.exploration_std) or not .01 <= self.exploration_std <= .5 or
                not math.isfinite(self.budget_seconds) or not 0 < self.budget_seconds <= 300):
            raise ValueError("Invalid seed/rate/preparation budget")


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def training_sampling_probabilities(episode_ids, train_indices):
    """Equal episode mass; split startup/steady mass within long episodes.

    Short student-visited episodes must not be drowned out by 600-step oracle
    trajectories. Held-out episodes never receive sampling mass.
    """
    import numpy as np
    ids, train = np.asarray(episode_ids), np.asarray(train_indices)
    probability = np.zeros(len(train), dtype=float)
    episodes = np.unique(ids[train])
    for episode in episodes:
        positions = np.flatnonzero(ids[train] == episode)
        all_rows = np.flatnonzero(ids == episode)
        startup = np.isin(train[positions], all_rows[:50])
        groups = [positions[startup], positions[~startup]]
        groups = [group for group in groups if len(group)]
        for group in groups:
            probability[group] = 1.0 / (len(episodes) * len(groups) * len(group))
    if not len(train) or not np.isfinite(probability).all() or np.any(probability <= 0):
        raise ValueError("Invalid episode-balanced training partition")
    return probability / probability.sum()


def validate_dataset(data, config):
    import numpy as np
    arrays = {name: np.asarray(data[name]) for name in
              ("observations", "privileged_observations", "actions", "episode_ids")}
    n = len(arrays["episode_ids"])
    if n < 32:
        raise PreparationDataInsufficient("Fewer than 32 reachable teacher samples")
    if n > config.episodes * config.steps_per_episode:
        raise ValueError("Unexpected teacher dataset size")
    for name, width in (("observations", 24), ("privileged_observations", 51), ("actions", 4)):
        value = arrays[name]
        if value.shape != (n, width) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"Invalid teacher array: {name}")
        arrays[name] = value.astype(np.float32)
    # Labels are actuator measurements. Silently clipping them would hide an
    # incorrect torque scale or a teacher that the student cannot represent.
    if np.any(np.abs(arrays["actions"]) > 1.0):
        raise ValueError("Teacher action/actual torque label exceeds the student range; no clipping allowed")
    ids = arrays["episode_ids"]
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= config.episodes):
        raise ValueError("Invalid teacher episode IDs")
    if len(np.unique(ids)) < 4:
        raise PreparationDataInsufficient("Fewer than four episodes contain reachable teacher samples")
    split_episodes(arrays["episode_ids"], config.seed)
    return arrays


def noise_profile(env):
    """Uniform sensor noise variance in the exact StudentEnv observation order."""
    import numpy as np
    cfg = env._config.noise_config
    level, scales = float(cfg.level), dict(cfg.scales)
    if not math.isfinite(level) or level < 0:
        raise ValueError("Invalid PPO observation noise")
    amplitude = np.zeros(24)
    for indices, name in ((slice(4, 7), "gyro"), (slice(7, 9), "pitch"), (9, "roll"),
                          (slice(10, 14), "joint_pos"), (slice(14, 18), "joint_vel"),
                          (slice(18, 20), "linvel")):
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
    privileged = np.zeros(51)
    privileged[:24] = variance
    return dict(level=level, scales=scales, command_ranges=command_ranges, actor_variance_prior=variance.tolist(),
                critic_variance_prior=privileged.tolist(),
                semantics="Uniform sensor noise and configured command-range variance added to train-only statistics; mean unchanged, summed_variance and std updated together")


def _identities(args, config, ppo_cfg):
    from force_teacher.contract import metadata, original_path
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
        sources["force_teacher/" + name] = _file_hash(Path(__file__).with_name(name))
    sources["wheel_teacher/teacher.py"] = _file_hash(Path(__file__).parents[1] / "wheel_teacher/teacher.py")
    # Native labels and replay depend on the actual virtual-force mapping and
    # student observation/reward code, even if the orchestration does not change.
    local_root = Path(__file__).resolve().parents[1]
    for root in (local_root / "hybrid", local_root / "legwheel", local_root / "assets"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                sources[path.relative_to(local_root).as_posix()] = _file_hash(path)
    packages = {name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "brax", "flax", "optax", "mujoco", "numpy")}
    data_identity = dict(schema=SCHEMA, contract={k: v for k, v in contract.items()
                                               if k not in ("source_fingerprint", "force_source_fingerprint")},
                         sources=sources, packages={k: packages[k] for k in ("mujoco", "numpy")},
                         episodes=config.episodes, steps_per_episode=config.steps_per_episode, seed=config.seed)
    network = json.loads(json.dumps(dict(ppo_cfg.network_factory)))
    bc_identity = dict(schema=SCHEMA, source_sha256=_file_hash(__file__), packages=packages,
                       ppo_adapter_sha256=_file_hash(Path(__file__).with_name("ppo.py")),
                       network_head_sha256=_file_hash(Path(__file__).with_name("networks.py")),
                       network=network, normalize_observations=bool(ppo_cfg.normalize_observations),
                       updates=config.updates, batch_size=config.batch_size, learning_rate=config.learning_rate,
                       exploration_std=config.exploration_std,
                       sampling="equal-episode-startup0.5s-steady-half-v1",
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


def _dataset(args, config, directory, expected, check, cancelled, monitor, policy_callable=None):
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
    from force_teacher.teacher import collect_dataset, CollectionCancelled
    def progress(event):
        check()
        monitor.publish(phase="preparing", stage="teacher_collect", preparation_message="采集教师实际腿力和轮扭矩（严格可达筛选）",
                        teacher_progress=event)
    try:
        data = collect_dataset(args.original_root, episodes=config.episodes, steps_per_episode=config.steps_per_episode,
                               seed=config.seed, on_progress=progress, cancelled=cancelled,
                               policy_callable=policy_callable,
                               student_probability=1.0 if policy_callable is not None else 0.0)
    except CollectionCancelled:
        check()  # Translate an explicit user stop into the console's StopRequested.
        raise
    # Retain an honest coverage report even when strict reachability filtering
    # leaves too little data to initialize a policy. This is not a success cache.
    _json(directory / "collection-report.json", dict(identity=expected, teacher_report=data["report"]))
    try:
        arrays = validate_dataset(data, config)
    except PreparationDataInsufficient as exc:
        _json(Path(args.run_dir) / "pretraining-report.json", dict(schema=SCHEMA, completed=False,
              reason=str(exc), teacher=data["report"], scope="Insufficient reachable teacher coverage; no BC parameters produced"))
        raise
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
    from force_teacher.networks import make_ppo_networks
    from brax.training.acme import running_statistics
    train, heldout, partition = split_episodes(arrays["episode_ids"], config.seed)
    observations = dict(state=jp.asarray(arrays["observations"]),
                        privileged_state=jp.asarray(arrays["privileged_observations"]))
    sizes = {key: value.shape[1] for key, value in observations.items()}
    factory = make_ppo_networks(sizes, 4, preprocess_observations_fn=running_statistics.normalize,
                                         **dict(ppo_cfg.network_factory))
    state = running_statistics.init_state({key: jax.ShapeDtypeStruct((size,), np.float32) for key, size in sizes.items()})
    normalizer = running_statistics.update(state, {key: value[train] for key, value in observations.items()})
    prior = dict(state=jp.asarray(noise["actor_variance_prior"]),
                 privileged_state=jp.asarray(noise["critic_variance_prior"]))
    variance = {key: normalizer.summed_variance[key] + len(train) * prior[key] for key in observations}
    normalizer = normalizer.replace(summed_variance=variance,
        std={key: jp.sqrt(jp.maximum(variance[key] / len(train), 1e-12)) for key in observations})
    from force_teacher.ppo import fixed_normalizer
    normalizer = fixed_normalizer(normalizer)
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
    """The checked 24->64->64->8 swish/tanh actor; no simulator/GPU calls."""
    import numpy as np
    mean, std = np.asarray(params[0].mean["state"]), np.asarray(params[0].std["state"])
    layers = sorted(params[1]["params"].items())
    if mean.shape != (24,) or std.shape != (24,) or np.any(std <= 0) or len(layers) != 3:
        raise ValueError("Unsupported BC actor layout")
    weights = []
    for (_, layer), shape in zip(layers, ((24, 64), (64, 64), (64, 8)), strict=True):
        kernel, bias = np.asarray(layer["kernel"]), np.asarray(layer["bias"])
        if kernel.shape != shape or bias.shape != (shape[1],):
            raise ValueError("Unsupported BC actor layer")
        weights.append((kernel, bias))
    def policy(observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (24,) or not np.isfinite(x).all():
            raise ValueError("Invalid student observation")
        x = (x - mean) / std
        for index, (kernel, bias) in enumerate(weights):
            x = x @ kernel + bias
            if index < 2:
                x = x / (1 + np.exp(-np.clip(x, -80, 80)))
        from force_teacher.networks import transform_logits
        action = np.tanh(transform_logits(x, np.asarray(observation, dtype=np.float32), xp=np)[:4])
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
    output["kernel"] = output["kernel"].at[:, 2:4].set(0)
    output["bias"] = output["bias"].at[2:4].set(0)
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
    sample_probability = training_sampling_probabilities(arrays["episode_ids"], train)
    final_loss = initial_train_loss
    for step in range(config.updates):
        check()
        indices = jp.asarray(generator.choice(train, size=config.batch_size, replace=True, p=sample_probability))
        actor, opt_state, value = update(actor, opt_state, indices)
        if step == 0 or (step + 1) % 16 == 0 or step + 1 == config.updates:
            final_loss = float(value)
            if not math.isfinite(final_loss):
                raise FloatingPointError("Nonfinite BC loss")
            monitor.publish(phase="preparing", stage="teacher_fit", bc_updates=step + 1,
                            bc_total_updates=config.updates, bc_train_loss=final_loss,
                            preparation_message="拟合可达教师腿力和轮扭矩")
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
                  heldout_leg_force_nrmse=float(np.sqrt(np.mean((after[:, :2]-held_targets[:, :2])**2))),
                  heldout_leg_force_rmse_n=np.sqrt(np.mean(((after[:, :2]-held_targets[:, :2]) *
                      np.asarray(noise["physical_action_scales"][:2]))**2, axis=0)).tolist(),
                  heldout_wheel_torque_rmse_nm=np.sqrt(np.mean(((after[:, 2:]-held_targets[:, 2:]) * .0295)**2, axis=0)).tolist(),
                  heldout_wheel_torque_nrmse=float(np.sqrt(np.mean((after[:, 2:]-held_targets[:, 2:])**2))),
                  numpy_actor_max_error=host_error, **partition,
                  initial_exploration_pre_tanh_std=config.exploration_std,
                  training_sampler="Equal mass per train episode; first 0.5 seconds and remaining samples each receive half within long episodes",
                  exploration_semantics="Zero std-head kernels with inverse-softplus bias; trainable again during PPO",
                  raw_actor_min_std=float(np.std(arrays["observations"][train], axis=0).min()),
                  raw_actor_low_variance_dimensions=np.flatnonzero(np.std(arrays["observations"][train], axis=0) < 1e-5).tolist(),
                  fitted_actor_min_std=float(np.asarray(normalizer.std["state"]).min()),
                  normalization_noise_prior=noise,
                  normalizer_scope="Training episodes only; held-out episodes never fit normalization or gradients",
                  heldout_scope="Episode split on one fixed model and a small command schedule; not cross-randomization generalization",
                  actor_initialization="Fresh 24D student actor; leg force and wheel torque means fitted from reachable measured teacher effort",
                  loss_semantics="Mean squared force/Fscale and wheel torque/0.0295 errors, equal weight per normalized actuator; never mixes raw N with raw Nm",
                  critic_initialization="Random untrained value network, fitted observation normalization only",
                  scope="Short BC initialization, not stable-balance or sim-to-real acceptance")
    return params, payload, report


def _load_params(payload, arrays, config, ppo_cfg, noise):
    from flax import serialization
    _, template, _, _, _, _ = _networks(arrays, config, ppo_cfg, noise)
    params = serialization.from_bytes(template, payload)
    parameter_digest(params)
    return params


def closed_loop_gate(replay, config):
    """Reject missing, shortened, invalid, or unsuccessful physical evidence."""
    durations = replay.get("student_episode_seconds", [])
    episodes = replay.get("episodes", [])
    passed = (replay.get("gate_passed") is True and len(durations) == config.replay_episodes
              and len(episodes) == config.replay_episodes
              and all(isinstance(value, (int, float)) and math.isfinite(value)
                      and value >= 6.0 for value in durations)
              and all(not episode.get("termination", {}).get("reasons", ["missing"])
                      for episode in episodes))
    return dict(passed=bool(passed), required_episodes=config.replay_episodes,
                required_seconds=6.0, student_episode_seconds=durations,
                criteria="Three standing episodes with pitch 0/+0.01/-0.01 rad reach six seconds without termination or invalid state")


BOOTSTRAP_SCHEMA = "force-verified-bootstrap-v1"
BOOTSTRAP_FILES = frozenset(("params.msgpack", "dataset.npz", "candidate-report.json",
                             "policy_metadata.json", "policy.cpp.txt", "validation.json"))


def _bootstrap_identity(args, env, ppo_cfg):
    from force_teacher.contract import metadata
    source_root = str(Path(__file__).resolve().parents[1])
    def portable(value):
        if isinstance(value, dict):
            return {str(key): portable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(item) for item in value]
        if isinstance(value, (str, Path)):
            return str(value).replace(source_root, "<source>")
        return value
    return dict(schema=BOOTSTRAP_SCHEMA, contract=metadata(args.original_root),
        environment=portable(env._config.to_dict()), network=portable(dict(ppo_cfg.network_factory)),
        packages={name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "brax", "flax", "optax", "mujoco", "numpy")},
        python=sys.version, exploration_pre_tanh_std=.01,
        normalization_policy="fixed_bc", evaluation_policy="deterministic",
        jax_environment={key: value for key, value in os.environ.items()
                         if key.startswith(("JAX_", "XLA_")) and key != "JAX_COMPILATION_CACHE_DIR"})


def _read_bootstrap(directory, identity, ppo_cfg):
    import numpy as np
    from types import SimpleNamespace
    record = _read(directory / "verified.json")
    if record.get("schema") != BOOTSTRAP_SCHEMA or record.get("completed") is not True or record["identity"] != identity:
        raise ValueError("Verified initializer identity changed")
    if record["bc_gate"].get("passed") is not True:
        raise ValueError("Initializer has no physical gate")
    if not BOOTSTRAP_FILES.issubset(record["files"]):
        raise ValueError("Initializer receipt is missing required evidence")
    for name, expected in record["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Initializer evidence must stay inside its cache directory")
        path = directory / name
        if path.is_symlink() or not path.is_file() or _file_hash(path) != expected:
            raise ValueError("Verified initializer evidence is damaged: " + name)
    with np.load(directory / "dataset.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    params = _load_params((directory / "params.msgpack").read_bytes(), arrays,
                         SimpleNamespace(seed=0), ppo_cfg, record["calibration"])
    if parameter_digest(params) != record["parameter_tree_sha256"]:
        raise ValueError("Verified initializer parameter tree changed")
    return params, record


def _bootstrap_result(args, monitor, directory, params, record):
    report = dict(schema=SCHEMA, completed=True, source="verified_external_skip_candidate",
        bootstrap_cache_hit=True, bootstrap_receipt=str(directory / "verified.json"),
        provenance=record["provenance"], bc_gate=record["bc_gate"], replay=record["replay"],
        bc=record["bc"], teacher=record["provenance"],
        training_profile="standing_v2", normalization_policy="fixed_bc", evaluation_policy="deterministic",
        initial_params_path=str(directory / "params.msgpack"), initial_params_sha256=record["files"]["params.msgpack"],
        dataset_path=str(directory / "dataset.npz"), parameter_tree_sha256=record["parameter_tree_sha256"],
        preparation_seconds=0.0, scope="Explicit external initializer verified against current complete contract, factory, export and physical standing gate")
    _json(Path(args.run_dir) / "pretraining-report.json", report)
    monitor.publish(stage="teacher_bc_ready", bc_gate_passed=True, bootstrap_cache_hit=True,
                    preparation_message="已复用经过当前版本闭环验证的初始化策略")
    return dict(params=params, report=report)


def verify_bootstrap(args, monitor, env, ppo_cfg, *, candidate_report, candidate_scripts):
    """Explicitly import a real experiment, revalidate current code, then publish.

    No old report is rewritten. Only the exploration scale head is reset; the
    deterministic actor must stay identical. This is never a default BC receipt.
    """
    import jax
    import jax.numpy as jp
    import numpy as np
    from flax import serialization
    from orbax import checkpoint as ocp
    from types import SimpleNamespace
    from force_teacher.contract import metadata
    from force_teacher.policy import ForcePolicy, export_params
    from force_teacher.teacher import evaluate_student
    from force_teacher.ppo import restore_fixed_normalizer, normalizer_max_abs_change
    from legwheel.deployment.extract import load
    original = _read(Path(candidate_report))
    if original.get("gate", {}).get("passed") is not True:
        raise ValueError("External candidate did not pass its recorded physical gate")
    for name, hash_name in (("params_path", "params_sha256"), ("dataset_path", "dataset_sha256")):
        if _file_hash(original[name]) != original[hash_name]:
            raise ValueError("External candidate artifact hash mismatch")
    with np.load(original["dataset_path"], allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if (arrays["observations"].shape != (original["total_samples"], 24)
            or arrays["privileged_observations"].shape != (original["total_samples"], 51)
            or arrays["actions"].shape != (original["total_samples"], 4)
            or len(np.unique(arrays["episode_ids"])) != original["episodes"]
            or any(not np.isfinite(value).all() for value in arrays.values())):
        raise ValueError("External candidate dataset contract is invalid")
    calibration = original["fit"]["normalization_noise_prior"]
    template_config = SimpleNamespace(seed=0)
    params = _load_params(Path(original["params_path"]).read_bytes(), arrays, template_config, ppo_cfg, calibration)
    if parameter_digest(params) != original["parameter_tree_sha256"]:
        raise ValueError("External candidate parameter tree mismatch")
    indices = np.linspace(0, len(arrays["episode_ids"])-1, 256, dtype=int)
    sample = arrays["observations"][indices]
    before = np.asarray([numpy_actor(params)(obs) for obs in sample])
    factory, _, _, _, _, _ = _networks(arrays, template_config, ppo_cfg, calibration)
    distribution = factory.parametric_action_distribution
    raw_std = math.log(math.expm1(.01 / float(distribution._var_scale) - float(distribution._min_std)))
    actor = jax.tree.map(lambda value: jp.asarray(value), params[1])
    output = actor["params"][sorted(actor["params"])[-1]]
    output["kernel"] = output["kernel"].at[:, 4:].set(0)
    output["bias"] = output["bias"].at[4:].set(raw_std)
    actor = jax.tree.map(lambda value: np.asarray(value).copy(), jax.device_get(actor))
    params = (params[0], actor, params[2])
    after = np.asarray([numpy_actor(params)(obs) for obs in sample])
    if not np.array_equal(before, after):
        raise ValueError("Exploration reset changed deterministic actions")
    logits = factory.policy_network.apply(params[0], params[1], {"state": jp.asarray(sample)})
    parity = float(np.max(np.abs(np.asarray(distribution.mode(logits))-after)))
    if parity > 1e-5:
        raise ValueError("Integrated Brax/NumPy policy mismatch")
    identity = _bootstrap_identity(args, env, ppo_cfg)
    provenance = dict(source="explicit_external_multiround_skip_candidate", candidate_report_sha256=_file_hash(candidate_report),
        candidate_report=str(candidate_report), candidate_params_sha256=original["params_sha256"],
        candidate_dataset_sha256=original["dataset_sha256"], episodes=original["episodes"], samples=original["total_samples"],
        fit_updates=original["updates"], scripts={str(path): _file_hash(path) for path in candidate_scripts},
        exploration_reset=dict(pre_tanh_std=.01, modified="Only std head kernels/bias; deterministic mean actions bitwise unchanged"))
    key = _hash_bytes(_canonical(dict(identity=identity, provenance=provenance)))
    cache = Path(getattr(args, "pretrain_cache_root", Path(args.torque_root) / "artifacts/force-teacher-pretrain"))
    directory = cache / "verified-bootstrap" / key
    monitor.check_stop()
    with _locked(cache / ".lock", monitor.check_stop):
        directory.mkdir(parents=True, exist_ok=True)
        _atomic(directory / "params.msgpack", serialization.to_bytes(params))
        _atomic(directory / "dataset.npz", Path(original["dataset_path"]).read_bytes())
        _json(directory / "candidate-report.json", original)
        _json(directory / "policy_metadata.json", metadata(args.original_root))
        source_files = []
        for index, source in enumerate(candidate_scripts):
            name = "source-" + str(index) + "-" + Path(source).name
            _atomic(directory / name, Path(source).read_bytes())
            source_files.append(name)
        evidence = directory / ("verification-" + uuid.uuid4().hex)
        evidence.mkdir()
        checkpoint = evidence / "checkpoint"
        ocp.PyTreeCheckpointer().save(checkpoint, params)
        recovered = load(checkpoint)
        recovered = (restore_fixed_normalizer(recovered[0]), recovered[1], recovered[2])
        if parameter_digest(recovered) != parameter_digest(params) or normalizer_max_abs_change(params[0], recovered[0]) != 0:
            raise ValueError("Integrated initializer save/load changed parameters")
        exported = directory / "policy.cpp.txt"
        export_params(checkpoint, exported)
        export_metadata = metadata(args.original_root)
        export_metadata["export_sha256"] = _file_hash(exported)
        _json(directory / "policy_metadata.json", export_metadata)
        policy = ForcePolicy(exported, directory / "policy_metadata.json", args.original_root)
        export_error = float(np.max(np.abs(np.asarray([policy.action(obs) for obs in sample])-after)))
        if export_error > 1e-5:
            raise ValueError("Integrated initializer export changed actions")
        monitor.publish(stage="teacher_gate", bc_gate_passed=False, preparation_message="验证已集成导出策略的三组6秒站立")
        observed = []
        def observed_action(observation):
            observed.append(np.asarray(observation).copy())
            return policy.action(observation)
        replay = evaluate_student(args.original_root, observed_action, episodes=3, steps_per_episode=600, seed=10000,
            cancelled=lambda: (Path(args.run_dir) / "stop.request").exists())
        cursor, tails = 0, []
        for teacher_seconds, student_seconds in zip(replay["teacher_episode_seconds"], replay["student_episode_seconds"], strict=True):
            cursor += round(teacher_seconds * 100)
            count = round(student_seconds * 100)
            tail = np.asarray(observed[cursor:cursor+count])[-100:]
            cursor += count
            tails.append(dict(last_observation_gyro_rad_s=tail[-1, 4:7].tolist(),
                last_observation_pitch_sin=float(tail[-1, 8]), last_observation_roll_sin=float(tail[-1, 9]),
                last_second_pitch_sin_rate=float((tail[-1, 8]-tail[0, 8]) / max((len(tail)-1)*.01, .01)),
                last_second_roll_sin_rate=float((tail[-1, 9]-tail[0, 9]) / max((len(tail)-1)*.01, .01)),
                last_second_max_abs_gyro_rad_s=np.max(np.abs(tail[:, 4:7]), axis=0).tolist()))
        if cursor != len(observed):
            raise ValueError("Physical replay observation accounting differs from the evaluator")
        replay["tail_motion"] = tails
        replay["tail_motion_scope"] = "Last policy observation precedes the final physics step by 0.01s; final pitch/roll are in episodes[].termination"
        gate = closed_loop_gate(replay, PreparationConfig())
        _json(directory / "validation.json", dict(factory_numpy_max_error=parity, export_max_error=export_error,
                                                 replay=replay, gate=gate))
        if not gate["passed"]:
            raise PreparationGateFailed("Integrated exported initializer did not pass the physical gate")
        record = dict(schema=BOOTSTRAP_SCHEMA, completed=True, identity=identity, provenance=provenance,
            calibration=calibration, parameter_tree_sha256=parameter_digest(params), bc=original["fit"],
            bc_gate=gate, replay=replay, factory_numpy_max_error=parity, export_max_error=export_error,
            files={name: _file_hash(directory / name) for name in sorted(BOOTSTRAP_FILES) + source_files})
        _json(directory / "verified.json", record)
    return _bootstrap_result(args, monitor, directory, params, record)


def prepare(args, monitor, env, ppo_cfg):
    """Return a compatible Brax (normalizer, actor, critic) initialization."""
    if getattr(args, "restore", None):
        return dict(params=None, report=dict(skipped=True, reason="Resume restores its checkpoint; BC is not rerun"))
    bootstrap_root = Path(getattr(args, "pretrain_cache_root", Path(args.torque_root) / "artifacts/force-teacher-pretrain")) / "verified-bootstrap"
    if bootstrap_root.is_dir():
        monitor.check_stop()
        identity = _bootstrap_identity(args, env, ppo_cfg)
        for receipt in sorted(bootstrap_root.glob("*/verified.json"), key=lambda path: path.stat().st_mtime, reverse=True):
            try:
                params, record = _read_bootstrap(receipt.parent, identity, ppo_cfg)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            return _bootstrap_result(args, monitor, receipt.parent, params, record)
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
    import numpy as np
    force_scales = np.asarray(env.force_scales, dtype=float)
    if force_scales.shape != (2,) or not np.isfinite(force_scales).all() or np.any(force_scales <= 0):
        raise ValueError("Invalid calibrated leg force scales")
    noise["physical_action_scales"] = force_scales.tolist() + [.0295, .0295]
    bc_id["normalization_noise_prior"] = noise
    cache = Path(getattr(args, "pretrain_cache_root", Path(args.torque_root) / "artifacts/force-teacher-pretrain"))
    history, aggregate, parent_policy = [], None, None
    with _locked(cache / ".lock", check):
        for round_index in range(config.dagger_rounds + 1):
            round_config = replace(config, seed=config.seed + round_index * 1000)
            round_data_id = dict(data_id, collection_seed=round_config.seed, dagger_round=round_index,
                                 student_probability=1.0 if round_index else 0.0)
            if history:
                round_data_id.update(parent_params_sha256=history[-1]["initial_params_sha256"],
                                     parent_dataset_sha256=_file_hash(history[-1]["dataset_path"]))
            data_key = _hash_bytes(_canonical(round_data_id))
            check()
            monitor.publish(phase="preparing", stage="teacher_collect", preparation_message="检查教师数据缓存")
            arrays, data_report, data_hit = _dataset(args, round_config, cache / "data" / data_key, round_data_id,
                                                     check, cancelled, monitor, parent_policy)
            if aggregate is not None:
                arrays["episode_ids"] = arrays["episode_ids"] + round_index * config.episodes
                arrays = {key: np.concatenate((aggregate[key], arrays[key]), axis=0) for key in arrays}
                merged = cache / "data" / data_key / "aggregated.npz"
                _save_dataset(merged, arrays)
                data_report = dict(data_report, dataset_sha256=_file_hash(merged),
                    teacher_report=dict(current=data_report["teacher_report"], prior=history[-1]["teacher"],
                        semantics="DAgger aggregate with distinct episode identities"))
            dataset_path = cache / "data" / data_key / ("aggregated.npz" if aggregate is not None else "dataset.npz")
            fit_config = replace(config, episodes=config.episodes * (round_index + 1))
            aggregate = arrays
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
                params = _load_params(params_file.read_bytes(), arrays, fit_config, ppo_cfg, noise)
                if parameter_digest(params) != saved["report"]["parameter_tree_sha256"]:
                    raise ValueError("BC parameter tree identity changed")
                bc_report = saved["report"]
                bc_hit = True
            except (OSError, ValueError, KeyError, TypeError):
                check()
                params, payload, bc_report = _fit(arrays, fit_config, ppo_cfg, check, monitor, noise)
                _atomic(params_file, payload)
                _json(receipt, dict(identity=bc_id, params_sha256=_file_hash(params_file), report=bc_report))
            check()  # BC is already committed; replay cancellation need not refit it.
            from force_teacher.teacher import evaluate_student, CollectionCancelled
            monitor.publish(phase="preparing", stage="teacher_gate", bc_gate_passed=False,
                            preparation_message="验证真实腿力站立：3组微扰初态各6秒")
            replay_identity = dict(schema=SCHEMA, bc_key=bc_key, params_sha256=_file_hash(params_file),
                                   episodes=config.replay_episodes, steps_per_episode=config.replay_steps,
                                   seed=config.seed + 10000)
            replay_key = _hash_bytes(_canonical(replay_identity))
            replay_directory = cache / "replay" / replay_key
            replay_file, replay_receipt = replay_directory / "report.json", replay_directory / "assessment.json"
            replay_hit = False
            try:
                saved = _read(replay_receipt)
                if (saved.get("completed") is not True or saved["identity"] != replay_identity or
                        saved["report_sha256"] != _file_hash(replay_file)):
                    raise ValueError("Replay evidence changed")
                replay = _read(replay_file)
                replay_hit = True
            except (OSError, ValueError, KeyError, TypeError):
                try:
                    replay = evaluate_student(args.original_root, numpy_actor(params), episodes=config.replay_episodes,
                                              steps_per_episode=config.replay_steps, seed=config.seed + 10000, cancelled=cancelled)
                except CollectionCancelled:
                    check()
                    raise
                check()
                _json(replay_file, replay)
                _json(replay_receipt, dict(completed=True, identity=replay_identity,
                                          report_sha256=_file_hash(replay_file)))
            check()
            gate = closed_loop_gate(replay, config)
            report = dict(schema=SCHEMA, completed=gate["passed"], bc_gate=gate,
                          training_profile="standing_v2", normalization_policy="fixed_bc",
                          evaluation_policy="deterministic", dagger_round=round_index,
                          data_cache_hit=data_hit, bc_cache_hit=bc_hit,
                          replay_cache_hit=replay_hit,
                          data_key=data_key, bc_key=bc_key, dataset_path=str(dataset_path),
                          initial_params_path=str(params_file), initial_params_sha256=_file_hash(params_file),
                          parameter_tree_sha256=bc_report["parameter_tree_sha256"], teacher=data_report["teacher_report"],
                          bc=bc_report, replay=replay, preparation_seconds=time.monotonic()-start,
                          scope="Physical standing gate is mandatory before PPO; finite trials do not establish broad robustness")
            history.append(report)
            report["attempts"] = [dict(dagger_round=item["dagger_round"], bc_gate=item["bc_gate"],
                                       data_key=item["data_key"], bc_key=item["bc_key"]) for item in history]
            _json(Path(args.run_dir) / "pretraining-report.json", report)
            monitor.publish(stage="teacher_bc_ready" if gate["passed"] else "teacher_gate_failed",
                            bc_gate_passed=gate["passed"],
                            pretraining_report=str(Path(args.run_dir) / "pretraining-report.json"),
                            preparation_message="闭环门禁通过，可以开始PPO" if gate["passed"] else "BC闭环未通过，禁止进入PPO",
                            bc_cache_hit=bc_hit, teacher_data_cache_hit=data_hit)
            if gate["passed"]:
                return dict(params=params, report=report)
            parent_policy = numpy_actor(params)
            if round_index == config.dagger_rounds:
                raise PreparationGateFailed("BC closed-loop gate failed; PPO was not started")
