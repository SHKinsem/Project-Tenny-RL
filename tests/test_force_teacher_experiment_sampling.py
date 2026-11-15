"""Option routing and measured reference labels, without ML fitting."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/torque200'))
from force_teacher import teacher as t
from force_teacher.options import normalize_options


class SamplingOptionsTests(unittest.TestCase):
    def test_initial_perturbation_off_is_exact_nominal_and_on_changes_only_seeded_cases(self):
        state=SimpleNamespace(data=SimpleNamespace(qpos=np.r_[np.zeros(3),1.,np.zeros(9)],qvel=np.zeros(12)))
        for episode in range(8):
            q,v=t._initial_coordinates(state,episode,np.random.default_rng(4),False)
            np.testing.assert_array_equal(q,state.data.qpos);np.testing.assert_array_equal(v,state.data.qvel)
        q,v=t._initial_coordinates(state,1,np.random.default_rng(4),True)
        self.assertGreater(q[5],0)
        a=t._initial_coordinates(state,4,np.random.default_rng(4),True)
        b=t._initial_coordinates(state,4,np.random.default_rng(4),True)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        self.assertGreater(np.max(np.abs(a[1])),0)

    def test_reference_dispatch_and_dagger_rejection(self):
        options=normalize_options(dict(teacher_source='reference',dagger_rounds=0,initial_state_perturbation=False))
        with patch.object(t,'_collect_reference',return_value='reference') as collect:
            self.assertEqual(t._collect_native('root',4,10,0,lambda e:None,experiment_config=options),'reference')
            self.assertEqual(collect.call_args.args[-1],options)
        with patch.object(t,'_worker') as worker:
            with self.assertRaises(ValueError):t.collect_dataset('root',experiment_config=options,policy_callable=lambda obs:obs,student_probability=1.)
            worker.assert_not_called()
            t.collect_dataset('root',experiment_config=options)
            self.assertEqual(worker.call_args.kwargs['experiment_config'],options)

    def test_reference_filters_actual_unreachable_torque_and_previous_history(self):
        class Author:
            def __init__(self,root):
                self.mujoco=object()
                self.data=SimpleNamespace(qpos=np.r_[np.zeros(3),1.,np.zeros(9)],qvel=np.zeros(12))
            def reset(self,command,*,qpos=None,qvel=None):
                self.index=0
                if qpos is not None:self.data.qpos[:]=qpos
                if qvel is not None:self.data.qvel[:]=qvel
            def action(self):return np.zeros(4)
            def step(self,action):
                torque=np.array([2. if self.index==0 else .2,.2,.001,-.001])
                self.mujoco.torques[:]=[torque.copy() for _ in range(4)]
                self.mujoco.jacobians[:]=[np.array([.1,.1]) for _ in range(4)]
                self.mujoco.valid[:]=[True]*4
                self.index+=1
                return dict(done=False)
        options=normalize_options(dict(teacher_source='reference',dagger_rounds=0,
            initial_state_perturbation=False,standing_only=False))
        meta=dict(teacher_sha256='teacher',teacher_checkpoint='checkpoint',teacher_schema='schema')
        with patch.dict(sys.modules,{'wheel_teacher.teacher':SimpleNamespace(AuthorTeacher=Author)}), \
             patch.object(t,'read_geometry',return_value=[]), \
             patch.object(t,'scan_geometry',return_value=[dict(force_scale_n=10.,jmax_bound=.1)]*2), \
             patch.object(t,'_observations',side_effect=lambda teacher,previous:(np.r_[np.zeros(20),previous/np.asarray(t.TORQUE_LIMITS)].astype(np.float32),np.zeros(51,np.float32))), \
             patch.object(t,'metadata',return_value=meta):
            result=t._collect_reference('root',4,4,0,lambda event:None,options)
        self.assertEqual(result['actions'].shape,(8,4))
        self.assertEqual(result['report']['rejected_current_cycles'],4)
        self.assertEqual(result['report']['rejected_prior_history_cycles'],4)
        np.testing.assert_allclose(result['actions'][:,:2],.2)
        np.testing.assert_allclose(result['observations'][:,-4:],np.tile([.2,.2,.001/.0295,-.001/.0295],(8,1)))
        self.assertEqual(result['report']['peak_teacher_actual_torque_nm'][0],2.)
        self.assertEqual(result['report']['episodes'][1]['command'][0],.04)


if __name__=='__main__':unittest.main()
