"""Local acceptance gate. --static-only never grants permission to train.

Run from torque200: python -m hybrid.validate --output results/hybrid-check.json
Requires the user's existing JAX/MuJoCo environment for the full check.
"""
import argparse
import math
import time

from hybrid.core import (HIP_RANGE, XML, TORQUE_LIMITS, read_geometry,
                         scan_geometry, source_fingerprint, write_json)


def static_audit():
    return dict(schema="hybrid-local-validation-v1", source_fingerprint=source_fingerprint(),
                runtime_validated=False, hip_range=list(HIP_RANGE),
                scan=scan_geometry(read_geometry()),
                scope="Pure source-model kinematics only; no compiled model or dynamics tested")


def runtime_checks(report):
    import functools
    import importlib.metadata
    import inspect
    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    from legwheel.assets import load_mj_model
    from legwheel.env import LegWheelRobot
    from hybrid.env import HybridLegWheelRobot, hybrid_config
    from mujoco_playground import wrapper
    from legwheel.randomization import domain_randomize

    # Strict dependency check, no --no-deps workaround or automatic installs.
    from packaging.requirements import Requirement
    packages = ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "playground", "flax")
    for package in packages:
        for text in importlib.metadata.requires(package) or ():
            req = Requirement(text)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            if req.specifier and importlib.metadata.version(req.name) not in req.specifier:
                raise RuntimeError(f"Dependency conflict: {package} requires {req}; fix the local environment explicitly")
    model = load_mj_model(XML, 0.0025)
    legs = read_geometry()
    active_names = [f"{p}_{s}_joint" for p in ("thigh", "wheel") for s in ("left", "right")]
    active_ids = [model.joint(n).id for n in active_names]
    dofs = [int(model.jnt_dofadr[i]) for i in active_ids]
    assert model.nu == 4 and model.nq == 13 and model.nv == 12
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
    assert list(model.actuator_trnid[:, 0]) == active_ids
    assert np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
    assert np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
    assert np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
    np.testing.assert_array_equal(model.actuator_gear[:, 0], np.ones(4))
    np.testing.assert_array_equal(model.actuator_gear[:, 1:], np.zeros((4, 5)))
    np.testing.assert_array_equal(model.actuator_gainprm[:, 0], np.ones(4))
    np.testing.assert_array_equal(model.actuator_biasprm, np.zeros_like(model.actuator_biasprm))
    assert not np.any(model.jnt_actfrclimited[active_ids]), "Unexpected joint-level force limiter"
    assert np.all(model.actuator_ctrllimited) and np.all(model.actuator_forcelimited)
    model.actuator_ctrlrange[:] = np.array([(-v, v) for v in TORQUE_LIMITS])
    model.actuator_forcerange[:] = model.actuator_ctrlrange

    max_l_error = max_j_error = max_fd_error = 0.0
    for leg in legs:
        hip = model.joint(f"thigh_{leg.side}_joint").id
        knee = model.joint(f"calf_{leg.side}_joint").id
        wheel = model.joint(f"wheel_{leg.side}_joint").id
        hq, kq, wq = (int(model.jnt_qposadr[j]) for j in (hip, knee, wheel))
        hv, kv, wv = (int(model.jnt_dofadr[j]) for j in (hip, knee, wheel))
        hs, cs = (model.site(f"{p}_{leg.side}_site").id for p in ("thigh", "wheel"))
        assert model.qpos0[hq] == leg.hip_ref and model.qpos0[kq] == leg.knee_ref
        eq = [i for i in range(model.neq) if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
              and model.eq_obj1id[i] == knee and model.eq_obj2id[i] == hip]
        assert len(eq) == 1 and model.eq_active0[eq[0]]
        np.testing.assert_allclose(model.eq_data[eq[0], :5], leg.poly, atol=0, rtol=0)
        for q in np.linspace(*HIP_RANGE, 81):
            for residual in (0.0, -0.001, 0.001):
                d = mujoco.MjData(model)
                d.qpos[hq], d.qpos[kq] = q, leg.coupled_knee(q)+residual
                # A nontrivial rigid-body pose: distance/J must not depend on it.
                d.qpos[:3] = [0.03, -0.07, 0.25]
                d.qpos[3:7] = [math.cos(.2), 0.0, math.sin(.2), 0.0]
                d.qpos[wq] = 1.3
                mujoco.mj_forward(model, d)
                r = d.site_xpos[cs]-d.site_xpos[hs]
                length = np.linalg.norm(r)
                jc, jh = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, d, jc, None, cs)
                mujoco.mj_jacSite(model, d, jh, None, hs)
                j = (r/length) @ ((jc-jh)[:, hv]+leg.slope(q)*(jc-jh)[:, kv])
                l_ours, j_ours, _ = leg.evaluate(q, d.qpos[kq])
                max_l_error = max(max_l_error, abs(length-l_ours))
                max_j_error = max(max_j_error, abs(j-j_ours))
                assert abs((r/length) @ (jc-jh)[:, wv]) < 1e-9
                if residual == 0:
                    eps = 1e-6
                    lengths = []
                    for shift in (-eps, eps):
                        dd = mujoco.MjData(model)
                        dd.qpos[hq], dd.qpos[kq] = q+shift, leg.coupled_knee(q+shift)
                        mujoco.mj_forward(model, dd)
                        lengths.append(np.linalg.norm(dd.site_xpos[cs]-dd.site_xpos[hs]))
                    max_fd_error = max(max_fd_error, abs((lengths[1]-lengths[0])/(2*eps)-j_ours))
        q = jp.linspace(*HIP_RANGE, 81)
        got = jax.jit(jax.vmap(lambda x: jp.asarray(leg.evaluate(x, trig=jp))))(q)
        expected = np.asarray([leg.evaluate(float(x)) for x in q])
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-5)
    assert max_l_error < 1e-7 and max_j_error < 1e-5 and max_fd_error < 1e-5

    # Static actual actuator output: gain randomization and force clipping.
    for gain in (0.9, 1.0, 1.1):
        model.actuator_gainprm[:, 0] = gain
        for action in (np.ones(4), -np.ones(4), np.zeros(4)):
            d = mujoco.MjData(model)
            d.ctrl[:] = action*np.array(TORQUE_LIMITS)
            mujoco.mj_forward(model, d)
            expected = np.clip(d.ctrl*gain, -np.array(TORQUE_LIMITS), TORQUE_LIMITS)
            np.testing.assert_allclose(d.qfrc_actuator[dofs], expected, atol=1e-8)
    model.actuator_gainprm[:, 0] = 1

    def cfg(delay=0):
        c = hybrid_config()
        c.noise_config.level = 0.0
        c.cmd_config.cmd_prob = 0.0
        c.action_delay_steps = delay
        return c

    envs = {m: HybridLegWheelRobot(m, cfg()) for m in
            ("virtual_leg_force", "matched_envelope", "direct_torque")}
    steps = {m: jax.jit(e.step) for m, e in envs.items()}
    states = {m: e.reset(jax.random.PRNGKey(3)) for m, e in envs.items()}
    # Preserve original reset and its 0.0004 rad equality residual; never project
    # qk every step to manufacture ideal-constraint agreement.
    initial_residual = max(float(np.max(np.abs(e.geometry(states[m].data)[2]))) for m, e in envs.items())
    old = LegWheelRobot(cfg())
    old_state, old_step = old.reset(jax.random.PRNGKey(3)), jax.jit(old.step)
    max_residual = max_vel_residual = max_length_vel_residual = 0.0
    max_equivalence_error = 0.0
    for n in range(20):
        a = jp.asarray([.01*math.sin(n*.2), -.01*math.cos(n*.2), .01, -.01])
        for mode, env in envs.items():
            cmd = a.at[:2].multiply(-1) if mode == "matched_envelope" else a
            states[mode] = steps[mode](states[mode], cmd)
            s = states[mode]
            assert s.obs["state"].shape == (24,) and s.obs["privileged_state"].shape == (51,)
            assert np.isfinite(np.asarray(s.obs["state"])).all() and np.isfinite(float(s.reward))
            assert not bool(s.done), "Short diagnostic terminated; inspect before training"
            sent, actual = np.asarray(s.info["hybrid_command_substeps"]), np.asarray(s.info["hybrid_actual_substeps"])
            np.testing.assert_allclose(s.obs["state"][-4:], sent.mean(axis=0)/TORQUE_LIMITS, atol=1e-6)
            np.testing.assert_allclose(s.metrics["reward/energy_cost"], np.mean((actual/TORQUE_LIMITS)**2), atol=1e-6)
            assert np.all(np.abs(actual) <= np.asarray(TORQUE_LIMITS)+1e-6)
            assert float(s.metrics["hybrid/out_of_range"]) == 0
            max_residual = max(max_residual, float(s.metrics["hybrid/equality_position_max_rad"]))
            max_vel_residual = max(max_vel_residual, float(s.metrics["hybrid/equality_velocity_max_rad_s"]))
            max_length_vel_residual = max(max_length_vel_residual, float(s.metrics["hybrid/length_velocity_residual_max_m_s"]))
        vf, me = states["virtual_leg_force"], states["matched_envelope"]
        err = float(np.max(np.abs(np.asarray(vf.data.qpos-me.data.qpos))))
        max_equivalence_error = max(max_equivalence_error, err)
        np.testing.assert_allclose(vf.data.qpos, me.data.qpos, atol=2e-6, rtol=0)
        np.testing.assert_allclose(vf.reward, me.reward, atol=1e-5, rtol=0)
        # New direct_torque has revised costs/range, but identical raw physics in-range.
        old_state = old_step(old_state, a)
        np.testing.assert_allclose(states["direct_torque"].data.qpos, old_state.data.qpos, atol=2e-6, rtol=0)
    # Provisional diagnostic tolerances, not a physical safety certificate.
    assert max_residual < .01, "Equality error exceeds 0.01 rad diagnostic tolerance"
    assert max_length_vel_residual < .05, "Leg-length residual velocity exceeds 0.05 m/s"
    delayed = HybridLegWheelRobot("virtual_leg_force", cfg(2))
    s, fn = delayed.reset(jax.random.PRNGKey(4)), jax.jit(delayed.step)
    for _ in range(2):
        s = fn(s, jp.ones(4)*.01)
        np.testing.assert_allclose(s.info["hybrid_command_substeps"], 0, atol=1e-7)
    s = fn(s, jp.ones(4)*.01)
    assert np.any(np.asarray(s.info["hybrid_command_substeps"]) != 0)
    # Outside envelope => no command; never silently clamp geometry.
    s = delayed.reset(jax.random.PRNGKey(4))
    s = s.replace(data=s.data.replace(qpos=s.data.qpos.at[delayed.hip_q[0]].set(HIP_RANGE[1]+.1)))
    s = fn(s, jp.ones(4))
    assert bool(s.done)
    np.testing.assert_allclose(s.info["hybrid_command_substeps"], 0, atol=1e-7)
    # Finite but off-constraint state and invalid action must fail closed too.
    for bad in ("knee", "action"):
        s = delayed.reset(jax.random.PRNGKey(4))
        if bad == "knee":
            s = s.replace(data=s.data.replace(qpos=s.data.qpos.at[delayed.knee_q[0]].add(.1)))
        s = fn(s, jp.ones(4) if bad == "knee" else jp.asarray([jp.inf, 0., 0., 0.]))
        assert bool(s.done) and bool(s.info["hybrid_invalid"])
        np.testing.assert_allclose(s.info["hybrid_command_substeps"], 0, atol=1e-7)

    # Exercise exactly the batched/randomized/reset wrapper used by train_ppo.
    required = {"episode_length", "action_repeat", "randomization_fn"}
    if not required.issubset(inspect.signature(wrapper.wrap_for_brax_training).parameters):
        raise RuntimeError("Installed playground wrapper API differs; inspect before training")
    for mode in envs:
        wrapped_cfg = hybrid_config()
        wrapped_cfg.noise_config.level = 0.0  # Keep default command probability.
        e = HybridLegWheelRobot(mode, wrapped_cfg)
        keys = jax.random.split(jax.random.PRNGKey(100), 2)
        batched = jax.jit(jax.vmap(e.reset))(keys)
        bstep = jax.jit(jax.vmap(e.step))
        batched = bstep(batched, jp.zeros((2, 4)))
        assert batched.obs["state"].shape == (2, 24)
        wrapped = wrapper.wrap_for_brax_training(e, episode_length=2, action_repeat=1,
            randomization_fn=functools.partial(domain_randomize, rng=keys))
        ws = jax.jit(wrapped.reset)(keys)
        wrapped_step = jax.jit(wrapped.step)
        def rollout(s, unused):
            s = wrapped_step(s, jp.ones((2, 4))*.001)
            return s, (s.obs["state"], s.reward, s.done, s.info["truncation"])
        ws, (observations, rewards, dones, truncations) = jax.jit(
            lambda s: jax.lax.scan(rollout, s, None, length=6))(ws)
        jax.block_until_ready(ws)
        assert observations.shape == (6, 2, 24)
        assert np.isfinite(np.asarray(observations)).all() and np.isfinite(np.asarray(rewards)).all()
        assert np.any(np.asarray(dones)) and np.any(np.asarray(truncations)), "Time-limit truncation/reset was not exercised"
        # First substep of an auto-reset episode must not inherit torque history.
        ws.info["command"] = jp.ones_like(ws.info["command"])*.123
        ws = ws.replace(done=jp.ones_like(ws.done))
        ws = wrapped_step(ws, jp.zeros((2, 4)))
        np.testing.assert_allclose(ws.info["hybrid_actual_substeps"], 0, atol=1e-7)
        np.testing.assert_allclose(ws.info["hybrid_action_rate"], 0, atol=1e-7)
        np.testing.assert_allclose(ws.info["command"], ws.info["hybrid_initial_command"], atol=1e-7)

    report.update(runtime_validated=True, scope="Local geometry / JIT-vmap-scan / training-wrapper plumbing; not balance or hardware",
                  runtime_packages={p: importlib.metadata.version(p) for p in
                                    ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "playground", "flax")},
                  wrapper_randomization_reset_checked=True,
                  model_parameters=dict(mass_kg=float(np.sum(model.body_mass)),
                                        active_armature=model.dof_armature[dofs].tolist(),
                                        active_damping=model.dof_damping[dofs].tolist(),
                                        active_frictionloss=model.dof_frictionloss[dofs].tolist()),
                  mujoco_version=mujoco.__version__, jax_version=jax.__version__,
                  devices=[str(d) for d in jax.devices()],
                  compiled_geometry_max_length_error_m=max_l_error,
                  compiled_geometry_max_jacobian_error=max_j_error,
                  compiled_finite_difference_error=max_fd_error,
                  initial_equality_residual_rad=initial_residual,
                  short_diagnostic_seconds=.1, max_equality_residual_rad=max_residual,
                  max_equality_velocity_residual_rad_s=max_vel_residual,
                  max_length_velocity_residual_m_s=max_length_vel_residual,
                  equivalence_qpos_max_error=max_equivalence_error)


def main():
    if not __debug__:
        raise RuntimeError("Do not disable assertions in the experimental acceptance pipeline")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--static-only", action="store_true")
    args = p.parse_args()
    start = time.monotonic()
    report = static_audit()
    try:
        if not args.static_only:
            runtime_checks(report)
    except Exception as exc:
        report.update(runtime_validated=False, error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-start)
        write_json(args.output, report)
        raise
    report["elapsed_seconds"] = time.monotonic()-start
    write_json(args.output, report)
    print("HYBRID_LOCAL_VALIDATION_OK" if report["runtime_validated"] else "HYBRID_STATIC_ONLY_NOT_TRAINING_READY")


if __name__ == "__main__":
    main()
