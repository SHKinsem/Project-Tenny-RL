import copy
import json
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics as rs
from force_teacher.ppo import _restore_fixed_mode
from legwheel.deployment.extract import load
from experiments.aligned_limit_ablation.networks import make_ppo_networks
from experiments.aligned_limit_ablation.export import policy_template
from native_eval.brax_policy import policy_from_brax
from native_eval import Evaluator
from experiments.balance_diagnostics.runner import write_json

work=Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
manifest=json.loads((work/'inputs/manifest-limit1.json').read_text())
params=load(manifest['initial_checkpoint'])
normalizer=_restore_fixed_mode(params[0])
with np.load('$HOME/projects/tenny-reward-ablation-20261116/evaluations/initial/policy-0-long.npz') as data:
    obs=np.concatenate([data[f'world_{world}_observation'][:600:3] for world in range(5)])
observation={'state':jp.asarray(obs),'privileged_state':jp.zeros((len(obs),51))}
factories=[make_ppo_networks({'state':24,'privileged_state':51},4,
    preprocess_observations_fn=rs.normalize,**manifest['ppo']['network_factory']) for _ in range(2)]
logits=[jax.jit(network.policy_network.apply)(normalizer,params[1],observation) for network in factories]
logits=[np.asarray(jax.device_get(value)) for value in logits]
assert np.array_equal(logits[0],logits[1])
sample=[]
for network,value in zip(factories,logits):
    sample.append(np.asarray(jax.device_get(network.parametric_action_distribution.sample(jp.asarray(value),jax.random.key(42)))))
assert np.array_equal(sample[0],sample[1])
common=np.asarray(manifest['actuator_contract']['policy_leg_force_scales'])
common_error=float(np.max(np.abs(sample[0][:,:2]*common-sample[1][:,:2]*common)))
assert common_error==0
template=policy_template(manifest['policy_metadata'],manifest['ppo'])
descriptor=json.loads((work/'inputs/initial-limit1.json').read_text())
descriptor['policy']=policy_from_brax((normalizer,params[1]),template)
native=np.asarray(Evaluator(descriptor).policy_actions(obs))
det=np.asarray(jp.tanh(jp.asarray(logits[0][:,:4])))
export_error=float(np.max(np.abs(det-native)))
assert export_error<1e-5,export_error
# Move actual trainable means, keeping the immutable BC anchor untouched.
probe=copy.deepcopy(params[1])
last=max(probe['params'],key=lambda name:int(name.rsplit('_',1)[1]))
probe['params'][last]['bias']=jp.asarray(probe['params'][last]['bias']).at[:2].add(2.)
probe_logits=jax.jit(factories[0].policy_network.apply)(normalizer,probe,observation)
probe_actions=np.asarray(jp.tanh(probe_logits[:,:4]))
assert np.any(np.abs(probe_actions[:,:2])>.8)
descriptor['policy']=policy_from_brax((normalizer,probe),template)
probe_native=np.asarray(Evaluator(descriptor).policy_actions(obs))
probe_export_error=float(np.max(np.abs(probe_actions-probe_native)))
assert probe_export_error<1e-5,probe_export_error
result=json.loads((work/'inputs/native-alignment.json').read_text())
result.update(random_common_request_verified=True,common_request_max_abs_error=common_error,
    random_policy_logits_max_abs_error=float(np.max(np.abs(logits[0]-logits[1]))),
    random_action_max_abs_error=float(np.max(np.abs(sample[0]-sample[1]))),
    sampled_states=len(obs),sample_seed=42,jax_native_max_abs_action_error=export_error,
    mutable_probe_jax_native_max_abs_action_error=probe_export_error,
    mutable_probe_max_abs_leg_action=float(np.max(np.abs(probe_actions[:,:2]))),
    initial_random_force_request_over_low_capacity_fraction=np.mean(np.abs(sample[0][:,:2])>.5,axis=0).tolist(),
    sampling_scope='Same state and random key: pre-cap physical requests identical. Requests above low-arm capacity are intentionally clipped differently.')
write_json(work/'inputs/alignment-validation.json',result)
print(json.dumps(result,indent=2),flush=True)
