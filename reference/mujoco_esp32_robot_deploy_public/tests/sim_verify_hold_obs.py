#!/usr/bin/env python3
"""Sim 支路接到共享核 HoldObs，与本仓库 vendor/legwheel 的 _get_obs 逐步对比。

  JAX_PLATFORMS=cpu .venv/bin/python tests/sim_verify_hold_obs.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "host"))
from controller.hold_obs import HoldObs  # noqa: E402
from policy.ckpt import DEPLOY_CKPT  # noqa: E402
from sim.io import command_from_state, from_mjx, scalar, wheels, yaw_from_xmat  # noqa: E402
from sim.paths import VENDOR, ensure_plant  # noqa: E402

OUT_JSON = HERE / "sim_verify_hold_obs_results.json"

ATOL = 1e-4
SEG_S = 1.0
SCHEDULE = [
    ("stand", [0.0, 0.0, -0.25, 0.0]),
    ("forward", [0.2, 0.0, -0.25, 0.0]),
    ("backward", [-0.2, 0.0, -0.25, 0.0]),
    ("yaw_left", [0.0, 1.5, -0.25, 0.0]),
    ("yaw_right", [0.0, -1.5, -0.25, 0.0]),
    ("stand2", [0.0, 0.0, -0.25, 0.0]),
]


def _fw_from_state(fw: HoldObs, env, state, turn_on: bool = False) -> dict:
    frame = from_mjx(env, state)
    cmd = command_from_state(state)
    return fw.step(cmd, frame=frame, turn_on=turn_on)


def _compare(fw_out: dict, env, state) -> dict:
    import numpy as np

    obs = np.asarray(state.obs["state"])
    err = {
        "dpsi": abs(fw_out["obs_dpsi"] - float(obs[7])),
        "ds_l": abs(fw_out["obs_ds_l"] - float(obs[11])),
        "ds_r": abs(fw_out["obs_ds_r"] - float(obs[13])),
        "s0": abs(fw_out["s0"] - scalar(state.info["position_start"])),
        "yaw0": abs(fw_out["yaw0"] - scalar(state.info["sinyaw_start"])),
        "slots_equal": abs(float(obs[11]) - float(obs[13])),
    }
    err["ok"] = all(v <= ATOL for k, v in err.items() if k != "ok")
    return err


def _make_env():
    ensure_plant()
    from legwheel import LegWheelRobot, default_config

    cfg = default_config()
    cfg.noise_config.level = 0.0
    cfg.push_config.enable = False
    cfg.cmd_config.cmd_prob = 0.0
    cfg.cmd_start_delay_s = 1e6
    cfg.cmd_config.command_change_interval = 10**6
    cfg.hold_error_clip.enable = True
    cfg.hold_error_clip.wheel_sinpos = 3.0
    cfg.hold_error_clip.sinyaw = 1.0
    return LegWheelRobot(cfg)


def _record(name: str, ok: bool, detail: dict) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def run_kinematic(env) -> list[dict]:
    import jax
    import jax.numpy as jp

    results = []
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng, False)
    fw = HoldObs(reset_on_fallen=False)
    fw_out = _fw_from_state(fw, env, state, turn_on=True)
    err = _compare(fw_out, env, state)
    results.append(_record("reset_matches_env", err["ok"], err))

    # 共模轮转只增 Δs
    q = state.data.qpos
    data = state.data.replace(qpos=q.at[9].add(1.0).at[12].add(1.0))
    info = dict(state.info)
    info["command"] = jp.zeros(4)
    obs = env._get_obs(data, info)
    q_l, q_r = wheels(env, data)
    fw2 = HoldObs(reset_on_fallen=False)
    fw2.position_start = scalar(info["position_start"])
    fw2.yaw_sin_zero = scalar(info["sinyaw_start"])
    fw2.hold_anchors_inited = True
    out = fw2.step([0, 0, 0, 0], q_l=q_l, q_r=q_r, yaw_rad=yaw_from_xmat(env, data))
    ds_env = float(obs["state"][11])
    results.append(_record("common_mode_ds",
                           abs(out["obs_ds_l"] - 1.0) < 1e-5
                           and abs(ds_env - 1.0) < 1e-4
                           and abs(float(obs["state"][11]) - float(obs["state"][13])) < 1e-6,
                           {"fw_ds": out["obs_ds_l"], "env_ds": ds_env}))

    # 差动轮转 Δs=0
    data_y = state.data.replace(qpos=q.at[9].add(1.0).at[12].add(-1.0))
    obs_y = env._get_obs(data_y, info)
    q_l, q_r = wheels(env, data_y)
    out_y = fw2.step([0, 0, 0, 0], q_l=q_l, q_r=q_r, yaw_rad=yaw_from_xmat(env, data_y))
    # fw2 anchors already at reset; after previous step position_start unchanged
    # but previous common-mode step didn't change fw2.position_start (zero cmd)
    # Wait, fw2.step for common_mode already ran and didn't refresh (zero cmd).
    # Then this second step uses same fw2 - position_start still reset value.
    # But hold_position of differential is same as reset if original q_L,q_R plus ±1.
    # original s + 0.5*(1 + -1) = original s, ds=0. Good.
    # PROBLEM: first kinematic step already called fw2.step with q_l+1,q_r+1 so
    # fw2.position_start still reset, but we mutated fw2 by that step. Fine.
    # Second step: q is original+1,-1 relative to RESET qpos not relative to previous.
    # We pass data_y from original state q, not from data. Good.
    results.append(_record("differential_ds_zero",
                           abs(out_y["obs_ds_l"]) < 1e-5
                           and abs(float(obs_y["state"][11])) < 1e-4,
                           {"fw_ds": out_y["obs_ds_l"], "env_ds": float(obs_y["state"][11])}))

    # 只打 vx 不刷朝向锚点
    far_y = float(env._getSinYaw_newest(state.data)) + 1.0
    info_vx = dict(state.info)
    info_vx["command"] = jp.array([0.2, 0.0, 0.0, 0.0])
    info_vx["sinyaw_start"] = jp.array(far_y)
    state_vx = env.step(state.replace(info=info_vx), jp.zeros(env.action_size))
    fw_vx = HoldObs(reset_on_fallen=False)
    fw_vx.hold_anchors_inited = True
    fw_vx.position_start = scalar(info_vx["position_start"])
    fw_vx.yaw_sin_zero = far_y
    out_vx = _fw_from_state(fw_vx, env, state_vx)
    err_vx = _compare(out_vx, env, state_vx)
    frozen = abs(scalar(state_vx.info["sinyaw_start"]) - far_y) < 1e-5
    results.append(_record("vx_freezes_heading",
                           frozen and err_vx["ok"],
                           {**err_vx, "yaw0": scalar(state_vx.info["sinyaw_start"]), "far_y": far_y}))

    # 只打 yaw 不刷位置锚点
    far_s = float(env.get_hold_position(state.data)) + 3.0
    info_yaw = dict(state.info)
    info_yaw["command"] = jp.array([0.0, 0.5, 0.0, 0.0])
    info_yaw["position_start"] = jp.array(far_s)
    state_yaw = env.step(state.replace(info=info_yaw), jp.zeros(env.action_size))
    fw_yaw = HoldObs(reset_on_fallen=False)
    fw_yaw.hold_anchors_inited = True
    fw_yaw.position_start = far_s
    fw_yaw.yaw_sin_zero = scalar(info_yaw["sinyaw_start"])
    out_yaw = _fw_from_state(fw_yaw, env, state_yaw)
    err_yaw = _compare(out_yaw, env, state_yaw)
    frozen_s = abs(scalar(state_yaw.info["position_start"]) - far_s) < 1e-5
    results.append(_record("yaw_freezes_position",
                           frozen_s and err_yaw["ok"],
                           {**err_yaw, "s0": scalar(state_yaw.info["position_start"]), "far_s": far_s}))

    # clip：锚点不动，观测被夹
    far_s2 = float(env.get_hold_position(state.data)) - 8.0
    info_c = dict(state.info)
    info_c["command"] = jp.zeros(4)
    info_c["position_start"] = jp.array(far_s2)
    obs_c = env._get_obs(state.data, info_c)
    fw_c = HoldObs(reset_on_fallen=False)
    fw_c.hold_anchors_inited = True
    fw_c.position_start = far_s2
    fw_c.yaw_sin_zero = scalar(info_c["sinyaw_start"])
    q_l, q_r = wheels(env, state.data)
    out_c = fw_c.step([0, 0, 0, 0], q_l=q_l, q_r=q_r, yaw_rad=yaw_from_xmat(env, state.data))
    env_ds = float(obs_c["state"][11])
    clip_ok = (
        abs(scalar(info_c["position_start"]) - far_s2) < 1e-6
        and abs(out_c["ds_true"]) > 5.0
        and abs(out_c["obs_ds_l"]) <= 5.0 + 1e-6
        and abs(env_ds) <= 5.0 + 1e-5
        and abs(out_c["obs_ds_l"] - env_ds) < 1e-4
    )
    results.append(_record("clip_obs_not_anchor", clip_ok, {
        "fw_ds_obs": out_c["obs_ds_l"], "fw_ds_true": out_c["ds_true"],
        "env_ds_obs": env_ds, "s0": far_s2,
    }))
    return results


def run_closed_loop(env) -> tuple[list[dict], dict]:
    import jax
    import jax.numpy as jp
    import numpy as np
    from policy.rl_policy import RL_Policy

    dt = float(env.dt)
    seg_steps = int(round(SEG_S / dt))
    n_steps = seg_steps * len(SCHEDULE)
    fw_pol = RL_Policy()
    jit_reset = jax.jit(lambda rng: env.reset(rng, False))
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(1)
    state = jit_reset(rng)
    fw = HoldObs(reset_on_fallen=False)
    _fw_from_state(fw, env, state, turn_on=True)

    max_err = {k: 0.0 for k in ("dpsi", "ds_l", "ds_r", "s0", "yaw0", "slots_equal")}
    n_ok = 0
    n_fail = 0
    n_reset = 0
    series = []
    cmd_series = []

    for step in range(n_steps):
        seg = step // seg_steps
        label, cmd = SCHEDULE[seg]
        if step % seg_steps == 0:
            state = state.replace(
                info={**state.info, "command": jp.array(cmd, dtype=jp.float32)}
            )
            print(f"[sim] t={step * dt:5.2f}s {label} {cmd}", flush=True)
        act = jp.asarray(
            fw_pol.run(np.asarray(state.obs["state"]).reshape(-1)), dtype=jp.float32
        )
        state = jit_step(state, act)
        done = bool(np.asarray(state.done).reshape(-1)[0])
        if done:
            n_reset += 1
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
            state = state.replace(
                info={**state.info, "command": jp.array(cmd, dtype=jp.float32)}
            )
            fw_pol.clean_history()
            fw = HoldObs(reset_on_fallen=False)
            _fw_from_state(fw, env, state, turn_on=True)
            continue
        fw_out = _fw_from_state(fw, env, state)
        err = _compare(fw_out, env, state)
        for k in max_err:
            max_err[k] = max(max_err[k], err[k])
        if err["ok"]:
            n_ok += 1
        else:
            n_fail += 1
        if step % 10 == 0:
            series.append({
                "t": step * dt,
                "ds_fw": fw_out["obs_ds_l"],
                "ds_env": float(np.asarray(state.obs["state"])[11]),
                "dpsi_fw": fw_out["obs_dpsi"],
                "dpsi_env": float(np.asarray(state.obs["state"])[7]),
                "label": label,
            })
            cmd_series.append({"t": step * dt, "vx": cmd[0], "yaw": cmd[1], "label": label})

    summary = {
        "steps_compared": n_ok + n_fail,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_reset": n_reset,
        "max_err": max_err,
        "ok": n_fail == 0 and (n_ok + n_fail) > 0,
        "ckpt": DEPLOY_CKPT,
        "series": series,
        "cmd_series": cmd_series,
    }
    results = [_record("closed_loop_policy_vs_env", summary["ok"], {
        "steps_compared": summary["steps_compared"],
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_reset": n_reset,
        "max_err": max_err,
        "ckpt": DEPLOY_CKPT,
    })]
    return results, summary


def main() -> int:
    print(f"[sim] plant={VENDOR}", flush=True)
    print("[sim] building env...", flush=True)
    env = _make_env()
    print("[sim] kinematic probes...", flush=True)
    kin = run_kinematic(env)
    for r in kin:
        print(f"  {'PASS' if r['ok'] else 'FAIL'} {r['name']}", flush=True)

    print(f"[sim] closed-loop with shared policy {DEPLOY_CKPT} ...", flush=True)
    loop, summary = run_closed_loop(env)
    for r in loop:
        print(f"  {'PASS' if r['ok'] else 'FAIL'} {r['name']} {r['detail']}", flush=True)

    payload = {
        "atol": ATOL,
        "ckpt": DEPLOY_CKPT,
        "kinematic": kin,
        "closed_loop": loop,
        "closed_loop_summary": {
            k: v for k, v in summary.items() if k not in ("series", "cmd_series")
        },
        "series": summary.get("series", []),
        "cmd_series": summary.get("cmd_series", []),
    }
    all_ok = all(r["ok"] for r in kin + loop)
    payload["all_ok"] = all_ok
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[sim] wrote {OUT_JSON}", flush=True)
    print("ALL PASS" if all_ok else "FAILED", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
