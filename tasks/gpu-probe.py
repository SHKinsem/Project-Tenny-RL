import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np

assert jax.default_backend() == 'gpu', jax.devices()
device = jax.devices()[0]
assert device.platform == 'gpu'
x = jax.device_put(np.ones((1024, 1024), dtype=np.float32), device)
f = jax.jit(lambda a: a @ a)
t0 = time.perf_counter()
y = f(x).block_until_ready()
first = time.perf_counter() - t0
t0 = time.perf_counter()
y = f(x).block_until_ready()
warm = time.perf_counter() - t0
np.testing.assert_allclose(np.asarray(y), 1024, rtol=0, atol=0)
report = dict(jax=jax.__version__, backend=jax.default_backend(),
              device=str(device), kind=device.device_kind,
              array_devices=[str(d) for d in y.devices()],
              compile_and_first_seconds=first, warm_seconds=warm,
              memory_stats=device.memory_stats())
out = Path('/home/jaccob/projects/tenny-reference/artifacts/gpu-probe.json')
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
