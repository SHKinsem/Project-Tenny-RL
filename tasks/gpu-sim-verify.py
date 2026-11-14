"""Run the original closed-loop checks on CUDA with separate output evidence."""
import importlib.util
from pathlib import Path
import jax

root = Path('/home/jaccob/projects/tenny-reference')
assert jax.default_backend() == 'gpu', jax.devices()
path = root / 'mujoco_esp32_robot_deploy_public/tests/sim_verify_hold_obs.py'
spec = importlib.util.spec_from_file_location('original_sim_verify', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT_JSON = root / 'artifacts/gpu-sim-verify.json'
print(f'GPU SIM devices: {jax.devices()}', flush=True)
raise SystemExit(module.main())
