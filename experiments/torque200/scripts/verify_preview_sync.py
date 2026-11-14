"""Read existing models; compare on-demand NPZ actor against Brax inference on CPU."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
from pathlib import Path
import tempfile
import json
import time
import numpy as np
ROOT = Path('/home/jaccob/projects/tenny-torque200')
WIN = Path('/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny/experiments/torque200')
sys.path[:0] = [str(WIN), str(ROOT / 'src')]
import jax
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics
from legwheel.deployment.extract import load
from preview import select_checkpoint
from preview_sync import SnapshotPublisher, read_snapshot
from training_control import TrainingController
controller = TrainingController(ROOT/'artifacts', Path('/home/jaccob/projects/tenny-reference'), ROOT)
results = {}
for mode in ('original', 'torque200'):
    model = controller.latest(mode)
    checkpoint = Path(model['checkpoint']) if model else select_checkpoint(ROOT/'artifacts')
    params = load(str(checkpoint))
    dims = {k: np.shape(v)[-1] for k,v in params[0].mean.items()}
    net = networks.make_ppo_networks(dims,4, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(64,64), value_hidden_layer_sizes=(512,512,512,512),
        policy_obs_key='state', value_obs_key='privileged_state')
    policy = jax.jit(networks.make_inference_fn(net)(params, deterministic=True))
    with tempfile.TemporaryDirectory(prefix='preview-numeric-') as directory:
        run = Path(directory)
        (run/'preview.request.json').write_text(json.dumps({'mode':mode,'request_id':'numeric-check'}))
        result = SnapshotPublisher(run,mode).service(123,params)
        assert 'error' not in result, result
        actor, metadata = read_snapshot(run/'preview-policy.npz',mode,'numeric-check')
        rng = np.random.default_rng(8)
        errors = []
        for i in range(32):
            obs = (params[0].mean['state'] + params[0].std['state'] * rng.uniform(-2,2,dims['state'])).astype(np.float32)
            expected = np.asarray(policy({'state':obs},jax.random.PRNGKey(0))[0])
            errors.append(float(np.max(np.abs(expected-actor.run(obs)))))
        assert max(errors) < 1e-5, errors
        results[mode] = dict(checkpoint=str(checkpoint), max_action_error=max(errors), observations=32,
                             snapshot_bytes=result['bytes'], publish_seconds=result['publish_seconds'])
output = WIN/'results/preview-on-open/numerical-check.json'
output.write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
