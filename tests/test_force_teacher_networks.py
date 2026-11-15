from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/torque200'))
from force_teacher.networks import transform_logits, WHEEL_RESIDUAL_SCALE, PREVIOUS_TORQUE_CLIP


class SkipHeadTests(unittest.TestCase):
    def test_batch_uses_raw_physical_previous_torque_and_keeps_other_logits(self):
        obs=np.zeros((3,24),np.float32);obs[:,22:24]=[[.2,-.3],[1.,-1.],[0.,0.]]
        logits=np.arange(24,dtype=np.float32).reshape(3,8)/10
        result=transform_logits(logits,obs)
        np.testing.assert_array_equal(result[:,:2],logits[:,:2])
        np.testing.assert_array_equal(result[:,4:],logits[:,4:])
        expected=np.arctanh(np.clip(obs[:,22:24],-1+PREVIOUS_TORQUE_CLIP,1-PREVIOUS_TORQUE_CLIP))+WHEEL_RESIDUAL_SCALE*logits[:,2:4]
        np.testing.assert_array_equal(result[:,2:4],expected)
        self.assertTrue(np.isfinite(result).all())

    def test_zero_wheel_mean_head_holds_previous_absolute_torque(self):
        obs=np.zeros(24,np.float32);obs[-2:]=[.4,-.2]
        action=np.tanh(transform_logits(np.zeros(8,np.float32),obs)[:4])
        np.testing.assert_allclose(action[-2:],obs[-2:],atol=1e-7)

    def test_wrong_observation_contract_rejected(self):
        with self.assertRaises(ValueError):transform_logits(np.zeros(8),np.zeros(26))
        with self.assertRaises(ValueError):transform_logits(np.zeros(6),np.zeros(24))

    def test_numpy_and_actual_jax_transforms_match(self):
        try:
            import jax
            import jax.numpy as jp
        except ImportError:
            self.skipTest('JAX checked in WSL integration')
        rng=np.random.default_rng(3)
        obs=rng.normal(size=(32,24)).astype(np.float32)
        logits=rng.normal(size=(32,8)).astype(np.float32)
        actual=jax.jit(lambda x,y:transform_logits(x,y,xp=jp))(jp.asarray(logits),jp.asarray(obs))
        np.testing.assert_allclose(np.asarray(actual),transform_logits(logits,obs),atol=1e-6)


if __name__=='__main__':unittest.main()
