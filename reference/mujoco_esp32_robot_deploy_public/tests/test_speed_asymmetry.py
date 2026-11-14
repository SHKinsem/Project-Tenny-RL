#!/usr/bin/env python3
"""把「前进快、右转快」拆成：映射 / 观测拼装 / 策略前向 / 本体。

问题：闭环里 |vx|=0.2 前进约是后退的 8 倍，|yaw|=0.5 右转约是左转的 1.8 倍。
这里分层看是中间代码把一边写小了，还是同一份网络本来就不奇对称。

  python tests/test_speed_asymmetry.py -v
  JAX_PLATFORMS=cpu $RL/.venv/bin/python tests/test_speed_asymmetry.py -v
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))

from controller.hold_obs import HoldObs
from controller.obs import ObsPack
from controller.sim_frame import SimFrame
from policy.rl_policy import RL_Policy, load_exported
from web.command import apply_control_page

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _has_sim() -> bool:
    try:
        from sim.paths import ensure_plant

        ensure_plant()
        from legwheel import LegWheelRobot, default_config  # noqa: F401

        return True
    except Exception:
        return False


HAS_SIM = _has_sim()


def _cm(act) -> float:
    a = np.asarray(act, dtype=float)
    return 0.5 * (a[2] + a[3])


def _df(act) -> float:
    a = np.asarray(act, dtype=float)
    return 0.5 * (a[3] - a[2])


def _ratio(a: float, b: float) -> float:
    return abs(a) / abs(b) if abs(b) > 1e-9 else float("inf")


def _stand_frame() -> SimFrame:
    return SimFrame(
        q_l=0.0,
        q_r=0.0,
        yaw_rad=0.0,
        pitch_rad=0.05,
        roll_rad=0.0,
        gyro=[0.0, 0.0, 0.0],
        thigh_l=-0.16,
        thigh_r=-0.16,
        thigh_vel_l=0.0,
        thigh_vel_r=0.0,
        wheel_vel_l=0.0,
        wheel_vel_r=0.0,
    )


def _pack_cmd(cmd, frame: SimFrame | None = None) -> list[float]:
    frame = frame or _stand_frame()
    hold = HoldObs()
    packer = ObsPack()
    hout = hold.step(list(cmd), frame, turn_on=True)
    return packer.pack(hout["cmd"], frame, hout)


class GlueMappingTests(unittest.TestCase):
    """人给的 ± 指令，映射必须奇对称。这里不对称才会是中间代码的锅。"""

    def test_keyboard_ws_is_odd(self) -> None:
        up = [0.0, 0.0, 0.0, 0.0]
        down = [0.0, 0.0, 0.0, 0.0]
        apply_control_page(0.0, 1.0, 50.0, 0.0, up)
        apply_control_page(0.0, -1.0, 50.0, 0.0, down)
        self.assertAlmostEqual(up[0], 0.2, places=6)
        self.assertAlmostEqual(down[0], -0.2, places=6)
        self.assertAlmostEqual(up[0] + down[0], 0.0, places=6)

    def test_forward_stick_left_right_yaw_is_odd(self) -> None:
        left = [0.0, 0.0, 0.0, 0.0]
        right = [0.0, 0.0, 0.0, 0.0]
        apply_control_page(-0.8, 0.2, 50.0, 0.0, left)
        apply_control_page(0.8, 0.2, 50.0, 0.0, right)
        self.assertAlmostEqual(left[1] + right[1], 0.0, places=5)
        self.assertGreater(left[1], 0.3)
        self.assertLess(right[1], -0.3)

    def test_action_scale_bias_are_left_right_equal(self) -> None:
        from controller.obs import ObsPackConfig

        cfg = ObsPackConfig()
        self.assertEqual(cfg.transform_action_scale[2], cfg.transform_action_scale[3])
        self.assertEqual(cfg.transform_action_bias[2], cfg.transform_action_bias[3])
        self.assertEqual(cfg.transform_action_scale[0], cfg.transform_action_scale[1])


class GlueObsTests(unittest.TestCase):
    """同一植物状态，只翻 command 符号，观测除 cmd 槽外必须一样。"""

    def test_vx_flip_only_changes_obs0(self) -> None:
        frame = _stand_frame()
        p = np.asarray(_pack_cmd([0.2, 0.0, -0.25, 0.0], frame))
        n = np.asarray(_pack_cmd([-0.2, 0.0, -0.25, 0.0], frame))
        self.assertAlmostEqual(p[0], 0.2, places=6)
        self.assertAlmostEqual(n[0], -0.2, places=6)
        np.testing.assert_allclose(p[1:], n[1:], atol=1e-7)

    def test_yaw_flip_only_changes_obs1(self) -> None:
        frame = _stand_frame()
        p = np.asarray(_pack_cmd([0.0, 0.5, -0.25, 0.0], frame))
        n = np.asarray(_pack_cmd([0.0, -0.5, -0.25, 0.0], frame))
        self.assertAlmostEqual(p[1], 0.5, places=6)
        self.assertAlmostEqual(n[1], -0.5, places=6)
        np.testing.assert_allclose(np.delete(p, 1), np.delete(n, 1), atol=1e-7)

    def test_hold_deadzone_is_even(self) -> None:
        hold = HoldObs()
        self.assertEqual(hold.cmd_lin_vel_x_zero_margin, 0.000195)
        self.assertEqual(hold.cmd_ang_vel_zero_margin, 0.01)

    def test_norm_cmd_slots_mean_near_zero(self) -> None:
        # (x-mean)/std 对 ±cmd 近似奇对称，不会单独把后退/左转吃掉。
        exported = load_exported()
        self.assertLess(abs(float(exported["mean"][0])), 0.01)
        self.assertLess(abs(float(exported["mean"][1])), 0.02)
        vx_p = (0.2 - exported["mean"][0]) / exported["std"][0]
        vx_n = (-0.2 - exported["mean"][0]) / exported["std"][0]
        yaw_p = (0.5 - exported["mean"][1]) / exported["std"][1]
        yaw_n = (-0.5 - exported["mean"][1]) / exported["std"][1]
        # mean 不是精确 0，归一化后 ±cmd 会剩约 0.006，远不够造成 8 倍速度差。
        self.assertLess(abs(float(vx_p + vx_n)), 0.02)
        self.assertLess(abs(float(yaw_p + yaw_n)), 0.02)


class PolicyFrozenTests(unittest.TestCase):
    """同一观测只翻 cmd 符号，看网络第一拍输出。这里不对称就是模型本身。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.exported = load_exported()
        cls.policy = RL_Policy()

    def _act(self, obs) -> np.ndarray:
        self.policy.clean_history()
        return np.asarray(self.policy.run(np.asarray(obs, dtype=np.float32)), dtype=float)

    def _mean_obs_with_cmd(self, vx: float, yaw: float) -> np.ndarray:
        obs = np.array(self.exported["mean"], dtype=float)
        obs[0] = vx
        obs[1] = yaw
        obs[2] = -0.25
        obs[3] = 0.0
        return obs

    def test_ckpt_at_training_mean_vx_is_odd(self) -> None:
        # 把观测钉在训练均值、只翻 vx：网络对前后几乎奇对称。8 倍差不在这一层。
        act_p = self._act(self._mean_obs_with_cmd(0.2, 0.0))
        act_n = self._act(self._mean_obs_with_cmd(-0.2, 0.0))
        cm_p, cm_n = _cm(act_p), _cm(act_n)
        print(
            f"\n[policy frozen mean-obs] vx±0.2  cm={cm_p:+.4f}/{cm_n:+.4f} "
            f"ratio={_ratio(cm_p, cm_n):.2f}  act+={np.round(act_p, 3)} act-={np.round(act_n, 3)}",
            flush=True,
        )
        self.assertLess(_ratio(cm_p, cm_n), 1.2)
        self.assertLess(abs(cm_p + cm_n), 0.05)

    def test_ckpt_differential_prefers_right_turn(self) -> None:
        # yaw>0 是左转（A），yaw<0 是右转（D）。差动 |右转| > |左转| → 网络偏右转。
        act_l = self._act(self._mean_obs_with_cmd(0.0, 0.5))
        act_r = self._act(self._mean_obs_with_cmd(0.0, -0.5))
        df_l, df_r = _df(act_l), _df(act_r)
        print(
            f"\n[policy frozen mean-obs] yaw±0.5  df={df_l:+.4f}/{df_r:+.4f} "
            f"|R|/|L|={_ratio(df_r, df_l):.2f}  actL={np.round(act_l, 3)} actR={np.round(act_r, 3)}",
            flush=True,
        )
        self.assertLess(df_r, 0.0)
        self.assertGreater(_ratio(df_r, df_l), 1.2)

    def test_packed_stand_obs_vx_is_not_odd(self) -> None:
        # 站立（轮速 0）离训练 wheel-vel 均值（约 -0.6/-0.3）很远。同一帧只翻 vx，
        # 共模残差不再奇对称 → 部署站立点上网络本身就偏。
        act_p = self._act(_pack_cmd([0.2, 0.0, -0.25, 0.0]))
        act_n = self._act(_pack_cmd([-0.2, 0.0, -0.25, 0.0]))
        act_l = self._act(_pack_cmd([0.0, 0.5, -0.25, 0.0]))
        act_r = self._act(_pack_cmd([0.0, -0.5, -0.25, 0.0]))
        print(
            f"\n[policy frozen packed stand] vx cm={_cm(act_p):+.4f}/{_cm(act_n):+.4f} "
            f"ratio={_ratio(_cm(act_p), _cm(act_n)):.2f}",
            flush=True,
        )
        print(
            f"[policy frozen packed stand] yaw df={_df(act_l):+.4f}/{_df(act_r):+.4f} "
            f"|R|/|L|={_ratio(_df(act_r), _df(act_l)):.2f}",
            flush=True,
        )
        self.assertGreater(_ratio(_cm(act_p), _cm(act_n)), 1.5)
        self.assertGreater(_ratio(_df(act_r), _df(act_l)), 1.1)


@unittest.skipUnless(HAS_SIM, "needs vendor plant + jax (sim venv)")
class ClosedLoopSplitTests(unittest.TestCase):
    """站稳后：开环恒动作测本体；闭环测策略平均动作。JIT 只做一次。"""

    env = None
    jit_reset = None
    jit_step = None

    @classmethod
    def setUpClass(cls) -> None:
        import jax
        from legwheel import LegWheelRobot, default_config
        from sim.paths import ensure_plant

        ensure_plant()
        cfg = default_config()
        cfg.noise_config.level = 0.0
        cfg.push_config.enable = False
        cfg.cmd_config.cmd_prob = 0.0
        cfg.hold_error_clip.enable = True
        cfg.hold_error_clip.wheel_sinpos = 3.0
        cfg.hold_error_clip.sinyaw = 1.0
        env = LegWheelRobot(cfg)
        cls.env = env
        # 必须 staticmethod：挂在 TestCase 上的函数会被当成实例方法，JIT 会吃到 self。
        cls.jit_reset = staticmethod(jax.jit(lambda rng: env.reset(rng, False)))
        cls.jit_step = staticmethod(jax.jit(env.step))
        cls.dt = float(env.dt)

    def _warmup(self, n: int = 80):
        import jax
        import jax.numpy as jp
        from sim.io import from_mjx

        policy, packer, hold = RL_Policy(), ObsPack(), HoldObs()
        state = self.jit_reset(jax.random.PRNGKey(0))
        stand = [0.0, 0.0, -0.25, 0.0]
        for i in range(n):
            info = dict(state.info)
            info["command"] = jp.asarray(stand)
            state = state.replace(info=info)
            frame = from_mjx(self.env, state)
            hout = hold.step(stand, frame, turn_on=(i == 0))
            obs = np.asarray(packer.pack(hout["cmd"], frame, hout), dtype=np.float32)
            act = policy.run(obs)
            packer.note_policy_action(act, frame)
            state = self.jit_step(state, jp.asarray(act, dtype=jp.float32))
        return state, policy, packer, hold

    def _drive_policy(self, cmd, n_warm=80, n_run=150):
        import jax.numpy as jp
        from sim.io import from_mjx

        state, policy, packer, hold = self._warmup(n_warm)
        q0 = np.asarray(state.data.qpos)
        yaw0 = from_mjx(self.env, state).yaw_rad
        acts, lins = [], []
        first = None
        for _ in range(n_run):
            info = dict(state.info)
            info["command"] = jp.asarray(cmd)
            state = state.replace(info=info)
            frame = from_mjx(self.env, state)
            hout = hold.step(cmd, frame, turn_on=False)
            obs = np.asarray(packer.pack(hout["cmd"], frame, hout), dtype=np.float32)
            act = np.asarray(policy.run(obs), dtype=float)
            if first is None:
                first = act.copy()
            acts.append(act)
            packer.note_policy_action(act, frame)
            state = self.jit_step(state, jp.asarray(act, dtype=jp.float32))
            lins.append(float(self.env.get_local_linvel_x(state.data)))
        q1 = np.asarray(state.data.qpos)
        yaw1 = from_mjx(self.env, state).yaw_rad
        dyaw = (yaw1 - yaw0 + math.pi) % (2.0 * math.pi) - math.pi
        A = np.asarray(acts)
        return {
            "dx": float(q1[0] - q0[0]),
            "dyaw": float(dyaw),
            "linvel": float(np.mean(lins)),
            "cm": float(np.mean(_cm(A))),
            "df": float(np.mean(_df(A))),
            "first": first,
        }

    def _wheel_vel_response(self, act, n_warm=80, n_run=30):
        """短开环：看轮速对残差的即时响应。整段位移会摔倒，不能用来比前后速度。"""
        import jax.numpy as jp
        from sim.io import from_mjx

        state, _, _, _ = self._warmup(n_warm)
        frame0 = from_mjx(self.env, state)
        w0 = 0.5 * (frame0.wheel_vel_l + frame0.wheel_vel_r)
        yaw0 = frame0.yaw_rad
        gz = []
        cmd = [0.0, 0.0, -0.25, 0.0]
        fixed = jp.asarray(act, dtype=jp.float32)
        for _ in range(n_run):
            info = dict(state.info)
            info["command"] = jp.asarray(cmd)
            state = state.replace(info=info)
            state = self.jit_step(state, fixed)
            fr = from_mjx(self.env, state)
            gz.append(fr.gyro[2])
        frame1 = from_mjx(self.env, state)
        w1 = 0.5 * (frame1.wheel_vel_l + frame1.wheel_vel_r)
        dyaw = (frame1.yaw_rad - yaw0 + math.pi) % (2.0 * math.pi) - math.pi
        return w1 - w0, dyaw, float(np.mean(gz))

    def test_plant_pid_wheel_residual_is_roughly_odd(self) -> None:
        # 0.3s 内轮速对 ±残差应近似反向。本体不是位移积分器，不能拿 dx 比前后。
        dw_p, _, _ = self._wheel_vel_response([0.0, 0.0, 0.25, 0.25])
        dw_n, _, _ = self._wheel_vel_response([0.0, 0.0, -0.25, -0.25])
        print(
            f"\n[plant PID 0.3s] ±0.25 cm  dω={dw_p:+.3f}/{dw_n:+.3f} "
            f"ratio={_ratio(dw_p, dw_n):.2f}",
            flush=True,
        )
        self.assertGreater(dw_p, 1.0)
        self.assertLess(dw_n, -1.0)
        self.assertLess(_ratio(dw_p, dw_n), 2.0)

    def test_plant_pid_differential_is_roughly_odd(self) -> None:
        _, yaw_l, gz_l = self._wheel_vel_response([0.0, 0.0, -0.25, 0.25])
        _, yaw_r, gz_r = self._wheel_vel_response([0.0, 0.0, 0.25, -0.25])
        print(
            f"\n[plant PID 0.3s] ±diff  dyaw={yaw_l:+.4f}/{yaw_r:+.4f} "
            f"gyro_z={gz_l:+.3f}/{gz_r:+.3f} |R|/|L|={_ratio(gz_r, gz_l):.2f}",
            flush=True,
        )
        self.assertGreater(gz_l, 0.0)
        self.assertLess(gz_r, 0.0)
        self.assertLess(_ratio(gz_r, gz_l), 2.0)

    def test_closed_loop_tracking_is_roughly_odd(self) -> None:
        fwd = self._drive_policy([0.2, 0.0, -0.25, 0.0])
        back = self._drive_policy([-0.2, 0.0, -0.25, 0.0])
        left = self._drive_policy([0.0, 0.5, -0.25, 0.0])
        right = self._drive_policy([0.0, -0.5, -0.25, 0.0])
        print(
            f"\n[closed-loop] W/S dx={fwd['dx']:+.4f}/{back['dx']:+.4f} "
            f"ratio={_ratio(fwd['dx'], back['dx']):.2f}  "
            f"mean_cm={fwd['cm']:+.4f}/{back['cm']:+.4f} "
            f"first_cm={_cm(fwd['first']):+.4f}/{_cm(back['first']):+.4f}",
            flush=True,
        )
        print(
            f"[closed-loop] A/D dyaw={left['dyaw']:+.4f}/{right['dyaw']:+.4f} "
            f"|R|/|L|={_ratio(right['dyaw'], left['dyaw']):.2f}  "
            f"mean_df={left['df']:+.4f}/{right['df']:+.4f} "
            f"first_df={_df(left['first']):+.4f}/{_df(right['first']):+.4f}",
            flush=True,
        )
        # 观测对齐训练 _get_obs / view.sh 之后，闭环位移应大致奇对称。
        self.assertLess(_ratio(fwd["dx"], back["dx"]), 2.5)
        self.assertLess(_ratio(right["dyaw"], left["dyaw"]), 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
