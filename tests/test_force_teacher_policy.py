"""Force-head semantics and precision must survive the actual text parser."""
from pathlib import Path
import json
import importlib.util
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments/torque200"))
from force_teacher import policy, contract
from force_teacher.networks import POLICY_HEAD_VERSION, WHEEL_RESIDUAL_SCALE, PREVIOUS_TORQUE_CLIP
_spec=importlib.util.spec_from_file_location("_force_test_kernel",Path(__file__).resolve().parents[1]/"experiments/torque200/legwheel/deployment/numpy_policy.py")
_kernel=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kernel)
load_exported=_kernel.load_exported


def parameters():
    stats = SimpleNamespace(mean={"state": np.linspace(-.123456789,.123456789,24,dtype=np.float32)},
                            std={"state": np.linspace(1e-7,.00123456789,24,dtype=np.float32)})
    actor={"params":{}}
    for i,shape in enumerate(((24,64),(64,64),(64,8))):
        actor["params"][f"hidden_{i}"]=dict(kernel=np.zeros(shape,dtype=np.float32),bias=np.zeros(shape[1],dtype=np.float32))
    actor["params"]["hidden_2"]["bias"][:4]=[.1,-.2,5.,-10.]
    return stats,actor,{}


class ForcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.export=self.root/"policy.cpp.txt"
        self.params=parameters()
        policy._write_export(self.params,self.export,"synthetic-unit-fixture")
        self.expected=dict(mode="leg_force_bc",teacher_schema=contract.SCHEMA,control_hz=100,
            policy_head=POLICY_HEAD_VERSION,wheel_residual_scale=WHEEL_RESIDUAL_SCALE,
            previous_torque_clip=PREVIOUS_TORQUE_CLIP,force_source_fingerprint="test-source")
        self.meta=self.root/"policy_metadata.json"
        self.meta.write_text(json.dumps({**self.expected,"export_sha256":contract.file_digest(self.export)}))

    def reader(self):
        with patch.object(policy,"metadata",return_value=self.expected), patch.dict(sys.modules,{"legwheel.deployment.numpy_policy":_kernel}):
            return policy.ForcePolicy(self.export,self.meta)

    def test_float32_stats_and_weights_round_trip_exactly(self):
        self.params[1]["params"]["hidden_0"]["kernel"][0,0]=np.float32(.123456789)
        policy._write_export(self.params,self.export,"precision")
        data=load_exported(self.export)
        np.testing.assert_array_equal(data["mean"],self.params[0].mean["state"])
        np.testing.assert_array_equal(data["std"],self.params[0].std["state"])
        np.testing.assert_array_equal(data["w1"],self.params[1]["params"]["hidden_0"]["kernel"].T)

    def test_skip_uses_raw_observation_and_unsquashed_logits(self):
        obs=np.zeros(24,dtype=np.float32);obs[22:24]=[.6,-.4]
        expected=np.tanh(np.r_[[.1,-.2],np.arctanh(obs[22:24])+.05*np.array([5.,-10.])])
        result=self.reader().action(obs)
        np.testing.assert_allclose(result,expected,atol=1e-7)
        np.testing.assert_array_equal(self.reader().run(obs),result)

    def test_previous_torque_extremes_and_invalid_observation(self):
        reader=self.reader()
        obs=np.zeros(24,dtype=np.float32);obs[22:24]=[1.,-1.]
        self.assertTrue(np.isfinite(reader.action(obs)).all())
        for bad in (np.zeros(26),np.full(24,np.nan)):
            with self.assertRaises(ValueError):reader.action(bad)

    def test_old_head_and_changed_export_are_rejected(self):
        saved=json.loads(self.meta.read_text());saved["policy_head"]="plain_tanh_v2"
        self.meta.write_text(json.dumps(saved))
        with self.assertRaises(ValueError):self.reader()
        saved["policy_head"]=POLICY_HEAD_VERSION;self.meta.write_text(json.dumps(saved))
        self.export.write_text(self.export.read_text()+"\n// changed bytes\n")
        with self.assertRaises(ValueError):self.reader()


if __name__=="__main__":unittest.main()
