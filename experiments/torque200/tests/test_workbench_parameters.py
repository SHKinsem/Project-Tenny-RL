import math
from pathlib import Path
import sys
import tempfile
import json
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench_parameters import (apply_parameters, default_parameters, parameter_schema,
                                  resolve_parameters, validate_parameters)


class ParameterTests(unittest.TestCase):
    def test_empty_edit_preserves_acceleration_and_other_modes(self):
        for mode in ("leg_force_bc", "original", "torque200"):
            effective, edits = resolve_parameters(mode)
            self.assertEqual(edits, {})
            self.assertEqual(effective, default_parameters(mode))
        self.assertEqual(default_parameters("leg_force_bc")["num_envs"], 512)
        self.assertEqual(default_parameters("original")["unroll_length"], 75)

    def test_reject_types_nonfinite_unknown_and_cross_field_shape(self):
        for values in ({"num_envs": True}, {"batch_size": 2.0}, {"learning_rate": "0.01"},
                       {"learning_rate": math.inf}, {"entropy_cost": math.nan}, {"seed": 1},
                       {"network_factory": {}}, {"num_eval_envs": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_parameters(values)
        with self.assertRaises(ValueError):
            resolve_parameters("leg_force_bc", overrides={"num_envs": 513})

    def test_edits_copy_config_and_validate_actual_shape(self):
        config = default_parameters("leg_force_bc")
        config["opaque"] = object()
        result = apply_parameters(config, {"num_envs": 256, "learning_rate": 0.00002})
        self.assertEqual(config["num_envs"], 512)
        self.assertIs(result["opaque"], config["opaque"])
        self.assertEqual(result["num_envs"], 256)
        self.assertEqual(result["learning_rate"], .00002)

    def test_resume_inherits_five_fields_and_previous_explicit_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / "checkpoints" / "000000000123"
            checkpoint.mkdir(parents=True)
            prior = default_parameters("leg_force_bc")
            prior.update(num_envs=128, batch_size=256, unroll_length=32)
            (run / "ppo_config.json").write_text(json.dumps(prior))
            effective, edits = resolve_parameters("leg_force_bc", checkpoint)
            self.assertEqual(effective["num_envs"], 128)
            self.assertEqual(effective["unroll_length"], 32)
            self.assertEqual(edits, {})
            (run / "request.json").write_text(json.dumps({"parameters": {"learning_rate": 1e-5}}))
            effective, edits = resolve_parameters("leg_force_bc", checkpoint, {"entropy_cost": .02})
            self.assertEqual(effective["learning_rate"], 1e-5)
            self.assertEqual(effective["entropy_cost"], .02)
            self.assertEqual(edits, {"learning_rate": 1e-5, "entropy_cost": .02})

    def test_locked_schema_uses_real_values_without_claiming_optimizer_restore(self):
        schema = parameter_schema("leg_force_bc", locked=True)
        self.assertEqual(schema["effective"]["num_envs"], 512)
        self.assertTrue(all(not field["editable"] for field in schema["fields"]))
        self.assertFalse(schema["resumeSemantics"]["optimizerRestored"])


if __name__ == "__main__":
    unittest.main()
