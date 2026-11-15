"""Checks the actual Brax fixed-EMA semantics, without an environment or training."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/torque200"))
from force_teacher.ppo import configure_ppo, fixed_normalizer, normalizer_max_abs_change, _restore_fixed_mode


class ConfigurationTests(unittest.TestCase):
    def test_fixed_scales_still_use_normalization(self):
        options = SimpleNamespace(normalize_observations=False)
        self.assertIs(configure_ppo(options), options)
        self.assertTrue(options.normalize_observations)
        self.assertEqual(options.normalize_observations_mode, 'ema')
        self.assertEqual(options.normalize_until_count, 0)
        self.assertTrue(options.deterministic_eval)


class RealStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import jax
            import jax.numpy as jp
            from brax.training.acme import running_statistics as rs
        except ImportError:
            raise unittest.SkipTest('Brax/JAX integration runs in the existing WSL venv')
        cls.jax, cls.jp, cls.rs = jax, jp, rs

    def state(self):
        values = self.jp.asarray([[1., 3.], [3., 7.], [2., 8.]])
        state = self.rs.init_state(self.jax.ShapeDtypeStruct((2,), np.float32))
        return fixed_normalizer(self.rs.update(state, values))

    def test_actual_update_preserves_all_scales_but_advances_count(self):
        state = self.state()
        updated = self.jax.jit(lambda value: self.rs.update(value,
            self.jp.asarray([[900., -800.], [-500., 700.]]), until_count=0))(state)
        self.assertEqual(normalizer_max_abs_change(state, updated), 0.)
        self.assertGreater(int(updated.count.lo), int(state.count.lo))
        x = self.jp.asarray([4., 2.])
        self.assertFalse(np.array_equal(np.asarray(self.rs.normalize(x, updated)), np.asarray(x)))
        np.testing.assert_array_equal(np.asarray(self.rs.normalize(x, state)),
                                      np.asarray(self.rs.normalize(x, updated)))

    def test_shared_factory_supports_real_ppo_checkpoint_network_config(self):
        from functools import partial
        from brax.training.agents.ppo import checkpoint
        from force_teacher.networks import make_ppo_networks
        factory = partial(make_ppo_networks, policy_hidden_layer_sizes=[64,64],
                          value_hidden_layer_sizes=[512]*4, policy_obs_key='state',
                          value_obs_key='privileged_state')
        config = checkpoint.network_config({'state':24,'privileged_state':51},4,True,factory)
        self.assertTrue(config.normalize_observations)
        self.assertEqual(list(config.network_factory_kwargs.policy_hidden_layer_sizes),[64,64])
        config.to_json_best_effort()

    def test_static_mode_lost_by_checkpoint_is_restored_without_scale_change(self):
        state = self.state()
        restored = _restore_fixed_mode(state.replace(mode=self.rs.NormalizationMode.WELFORD))
        self.assertEqual(restored.mode, self.rs.NormalizationMode.EMA)
        self.assertEqual(normalizer_max_abs_change(state, restored), 0.)
        updated = self.rs.update(restored, self.jp.zeros((10, 2)), until_count=0)
        self.assertEqual(normalizer_max_abs_change(state, updated), 0.)

    def test_welford_sum_variance_cannot_be_resumed_as_fixed_ema(self):
        state = self.state()
        with self.assertRaises(ValueError):
            _restore_fixed_mode(state.replace(summed_variance=state.summed_variance * 100))

    def test_changed_shape_or_nonfinite_stats_are_rejected(self):
        state = self.state()
        with self.assertRaises(ValueError):
            normalizer_max_abs_change(state, state.replace(std=self.jp.ones(3)))
        with self.assertRaises(ValueError):
            normalizer_max_abs_change(state, state.replace(std=self.jp.asarray([1., self.jp.nan])))

    def test_real_orbax_save_load_recovers_static_mode_and_uint64_tree(self):
        import tempfile
        from orbax import checkpoint as ocp
        from legwheel.deployment.extract import load
        from brax.training import checkpoint
        state = self.state()
        params = (state, {'weights': self.jp.ones((2, 4))}, {'weights': self.jp.zeros((2, 1))})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved'
            ocp.PyTreeCheckpointer().save(path, params)
            for restored in (load(path), checkpoint.load(path)):
                restored = (_restore_fixed_mode(restored[0]), restored[1], restored[2])
                self.assertEqual(self.jax.tree.structure(params), self.jax.tree.structure(restored))
                self.assertEqual(normalizer_max_abs_change(state, restored[0]), 0.)
                updated = self.rs.update(restored[0], self.jp.zeros((4, 2)), until_count=0)
                self.assertEqual(normalizer_max_abs_change(state, updated), 0.)


if __name__ == '__main__':
    unittest.main()
