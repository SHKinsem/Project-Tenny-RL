"""Real host-side regression tests; not JAX/MuJoCo or PPO validation."""
import copy
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np = None
try:
    import jax
except ImportError:
    jax = None

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "torque200"
sys.path.insert(0, str(ROOT))
from hybrid.checkpoints import check_network_update, network_diagnostics, network_pair


class BundleTest(unittest.TestCase):
    def test_only_known_outer_wrapper_is_canonicalized(self):
        policy = {"params": {"kernel": [1., 2.]}}
        value = {"params": {"bias": (3.,)}}
        for wrapper in (list, tuple):
            pair = network_pair(wrapper(("normalizer", policy, value)))
            self.assertIs(type(pair), tuple)
            self.assertIs(pair[0], policy)
            self.assertIs(pair[1], value)
            self.assertIs(type(pair[0]["params"]["kernel"]), list)
            self.assertIs(type(pair[1]["params"]["bias"]), tuple)

    def test_malformed_bundle_fails_closed(self):
        class ExtraList(list):
            pass

        class ExtraTuple(tuple):
            pass

        for params in (None, {}, [], [1, 2], (1, 2, 3, 4),
                       ExtraList([1, 2, 3]), ExtraTuple((1, 2, 3))):
            with self.subTest(params=params), self.assertRaises(AssertionError):
                network_pair(params)


@unittest.skipIf(np is None, "NumPy required for real host-array regression tests")
class NetworkUpdateTest(unittest.TestCase):
    def setUp(self):
        self.before = [dict(count=1),
                       {"params": {"a": np.array([[1., 2.]], dtype="float32"),
                                   "z": np.array([3.], dtype="float32")}},
                       {"params": {"bias": np.array([4.], dtype="float32")}}]
        self.after = tuple(copy.deepcopy(self.before))
        self.after[1]["params"]["a"][0, 0] += np.float32(0.25)

    def test_list_to_tuple_with_actual_update_passes(self):
        report = check_network_update(self.before, self.after)
        self.assertEqual(report["warm_restart_network_leaves_checked"], 3)
        self.assertEqual(report["warm_restart_changed_network_leaves"], 1)
        self.assertEqual(report["warm_restart_network_max_abs_change"], .25)
        self.assertEqual(report["warm_restart_outer_container_before"], "list")
        self.assertEqual(report["warm_restart_outer_container_after"], "tuple")

    def test_diagnostics_preserve_types_paths_shapes_dtypes_without_weights(self):
        report = network_diagnostics(self.before, self.after)
        self.assertEqual(report["before"]["bundle_type"], "list")
        self.assertEqual(report["after"]["bundle_type"], "tuple")
        leaf = report["before"]["leaves"][0]
        self.assertEqual(leaf["path"], ["policy", "params", "a"])
        self.assertEqual(leaf["shape"], [1, 2])
        self.assertEqual(leaf["dtype"], "float32")
        self.assertNotIn("values", leaf)
        self.assertIn("array_type", leaf)

    def test_dict_insertion_order_is_not_architecture(self):
        self.after[1]["params"] = dict(reversed(list(self.after[1]["params"].items())))
        self.assertEqual(check_network_update(self.before, self.after)["warm_restart_changed_network_leaves"], 1)

    def test_inner_list_to_tuple_remains_rejected(self):
        self.before[1]["nested"] = [np.array([1.])]
        self.after[1]["nested"] = (np.array([1.]),)
        with self.assertRaisesRegex(AssertionError, "paths or containers"):
            check_network_update(self.before, self.after)

    def test_missing_extra_and_renamed_paths_fail(self):
        for change in ("missing", "extra", "rename"):
            after = copy.deepcopy(self.after)
            if change == "missing":
                del after[2]["params"]["bias"]
            elif change == "extra":
                after[2]["params"]["another"] = np.array([4.], dtype="float32")
            else:
                after[2]["params"]["renamed"] = after[2]["params"].pop("bias")
            with self.subTest(change=change), self.assertRaises(AssertionError):
                check_network_update(self.before, after)

    def test_shape_change_is_not_a_weight_update(self):
        self.after[2]["params"]["bias"] = np.array([[4.]], dtype="float32")
        with self.assertRaisesRegex(AssertionError, "shape/dtype"):
            check_network_update(self.before, self.after)

    def test_dtype_change_is_not_a_weight_update(self):
        self.after[2]["params"]["bias"] = np.array([4.], dtype="float64")
        with self.assertRaisesRegex(AssertionError, "shape/dtype"):
            check_network_update(self.before, self.after)

    def test_all_leaves_checked_after_an_earlier_valid_change(self):
        for side in (self.before, self.after):
            for value in (np.nan, np.inf, -np.inf):
                original = side[2]["params"]["bias"][0]
                side[2]["params"]["bias"][0] = value
                with self.subTest(side=type(side), value=value), self.assertRaisesRegex(AssertionError, "Nonfinite"):
                    check_network_update(self.before, self.after)
                side[2]["params"]["bias"][0] = original

    def test_finite_but_unrepresentable_change_fails_closed(self):
        self.before[2]["params"]["bias"] = np.array([-1e308], dtype="float64")
        self.after[2]["params"]["bias"] = np.array([1e308], dtype="float64")
        with np.errstate(over="ignore"), self.assertRaisesRegex(AssertionError, "parameter change"):
            check_network_update(self.before, self.after)

    def test_unchanged_weights_rejected_even_with_normalization_changes(self):
        after = copy.deepcopy(self.before)
        after[0]["count"] += 99
        with self.assertRaisesRegex(AssertionError, "no changed"):
            check_network_update(self.before, after)

    def test_critic_only_change_preserves_existing_semantics(self):
        after = copy.deepcopy(self.before)
        after[2]["params"]["bias"][0] += 1
        self.assertEqual(check_network_update(self.before, after)["warm_restart_changed_network_leaves"], 1)

    def test_empty_network_rejected(self):
        self.before[2].clear()
        self.after[2].clear()
        with self.assertRaisesRegex(AssertionError, "Empty value"):
            check_network_update(self.before, self.after)

    def test_nonnumeric_leaf_rejected(self):
        self.after[2]["params"]["bias"] = np.array(["not a weight"])
        with self.assertRaisesRegex(AssertionError, "Non-floating"):
            check_network_update(self.before, self.after)

    def test_custom_array_projection_is_rejected(self):
        class Projection:
            hidden = float("nan")

            def __array__(self, dtype=None, copy=None):
                return np.array([4.], dtype=dtype or "float32")

        self.after[2]["params"]["bias"] = Projection()
        with self.assertRaisesRegex(AssertionError, "Unsupported"):
            check_network_update(self.before, self.after)
        self.assertIn("Unsupported", network_diagnostics(self.before, self.after)["after"]["diagnostic_error"])

    def test_custom_container_subclasses_are_rejected(self):
        class ExtraDict(dict):
            hidden = float("nan")

        class ExtraList(list):
            hidden = float("nan")

        class ExtraTuple(tuple):
            hidden = float("nan")

        for container in (ExtraDict(a=np.array([1.])), ExtraList([np.array([1.])]),
                          ExtraTuple((np.array([1.]),))):
            self.after[1]["custom"] = container
            with self.subTest(container=type(container)), self.assertRaisesRegex(AssertionError, "Unsupported"):
                check_network_update(self.before, self.after)

    def test_malformed_diagnostics_are_recorded_but_not_accepted(self):
        self.assertIn("diagnostic_error", network_diagnostics([], self.after)["before"])
        with self.assertRaises(AssertionError):
            check_network_update([], self.after)

    def test_nonstring_parameter_key_rejected(self):
        self.after[2]["params"][1] = np.array([1.])
        with self.assertRaisesRegex(AssertionError, "Non-string"):
            check_network_update(self.before, self.after)

    @unittest.skipIf(jax is None, "JAX unavailable; actual treedef/runtime regression not run")
    def test_actual_jax_treedef_list_tuple_regression_and_inner_guard(self):
        self.assertNotEqual(jax.tree.structure(self.before[1:]), jax.tree.structure(self.after[1:]))
        self.assertEqual(jax.tree.structure(network_pair(self.before)), jax.tree.structure(network_pair(self.after)))
        actual_jax_params = jax.tree.map(jax.numpy.asarray, self.after)
        self.assertIsInstance(actual_jax_params[1]["params"]["a"], jax.Array)
        self.assertEqual(check_network_update(self.before, actual_jax_params)["warm_restart_changed_network_leaves"], 1)
        self.before[1]["nested"] = [np.array([1.])]
        self.after[1]["nested"] = (np.array([1.]),)
        self.assertNotEqual(jax.tree.structure(network_pair(self.before)), jax.tree.structure(network_pair(self.after)))


if __name__ == "__main__":
    unittest.main()
