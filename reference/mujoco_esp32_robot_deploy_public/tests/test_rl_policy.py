"""共享策略核：从 src/policy/rl_policy.cpp 读数组，前向与固件同构。

运行：python tests/test_rl_policy.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from policy.ckpt import DEPLOY_CKPT, assert_shared_ckpt
from policy.rl_policy import LAYER_SIZES, RL_Policy, load_exported


class RLPolicyTests(unittest.TestCase):
    def test_framework_shares_one_ckpt(self) -> None:
        self.assertEqual(assert_shared_ckpt(), DEPLOY_CKPT)
        self.assertEqual(DEPLOY_CKPT, "ckpts/newobs-norecovery/000403660800")

    def test_export_is_newobs_norecovery(self) -> None:
        exported = load_exported()
        self.assertIn("newobs-norecovery/000403660800", exported["comment"])
        self.assertEqual(exported["mean"].size, LAYER_SIZES[0])
        # 双锚点：左右轮 Δs 槽同一分布。
        self.assertAlmostEqual(float(exported["mean"][11]), float(exported["mean"][13]), places=6)
        self.assertAlmostEqual(float(exported["std"][11]), float(exported["std"][13]), places=5)

    def test_forward_is_finite_and_bounded(self) -> None:
        policy = RL_Policy()
        act = policy.run(np.zeros(26, dtype=np.float32))
        self.assertEqual(act.shape, (4,))
        self.assertTrue(np.all(np.isfinite(act)))
        self.assertTrue(np.all(np.abs(act) <= 1.0 + 1e-5))

    def test_history_clean_resets_to_same_zero_action(self) -> None:
        policy = RL_Policy()
        a0 = policy.run(np.zeros(26, dtype=np.float32)).copy()
        policy.run(np.ones(26, dtype=np.float32))
        policy.clean_history()
        a1 = policy.run(np.zeros(26, dtype=np.float32))
        np.testing.assert_allclose(a0, a1, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
