"""Replace only this project's preview server; leave detached training untouched."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request
BASE = Path('/home/jaccob/projects/tenny-torque200')
WIN = Path('/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny/experiments/torque200')
PY = '/home/jaccob/projects/tenny-reference/.venv-gpu/bin/python'
web = str(BASE/'src/web_preview.py')
def processes(script):
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            argv = (proc/'cmdline').read_bytes().decode().split('\0')
        except (OSError,UnicodeError): continue
        if script in argv: found.append((int(proc.name), [x for x in argv if x]))
    return found
servers = processes(web)
assert len(servers) == 1, servers
pid, argv = servers[0]
assert '--port' in argv and argv[argv.index('--port')+1] == '8770', argv
training_before = processes(str(BASE/'src/console_train.py'))
for name in ('preview_sync.py','console_train.py','training_control.py','web_preview.py','dashboard.html'):
    shutil.copy2(WIN/name,BASE/'src'/name)
os.kill(pid,signal.SIGINT)
deadline = time.monotonic()+15
while processes(web):
    if time.monotonic()>deadline: raise RuntimeError('Old web server did not exit; training left untouched')
    time.sleep(.2)
env = os.environ.copy()
env.update(JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',JAX_COMPILATION_CACHE_DIR=str(BASE/'jax-cache'))
env.pop('MUJOCO_GL',None)
with (BASE/'artifacts/preview-on-open-web.log').open('ab') as log:
    child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=BASE/'src',start_new_session=True)
print('WEB_PID',child.pid,flush=True)
deadline=time.monotonic()+60
while time.monotonic()<deadline:
    if child.poll() is not None: raise RuntimeError('New web exited; inspect preview-on-open-web.log')
    try:
        state=json.load(urllib.request.urlopen('http://127.0.0.1:8770/state',timeout=2))
        if 'sync_status' in state:
            result=dict(web_pid=child.pid, mode=state['mode'], suspended=state['suspended'],
                        training_before=training_before,training_after=processes(str(BASE/'src/console_train.py')))
            (WIN/'results/preview-on-open/deployment.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)
            break
    except (OSError,ValueError): pass
    time.sleep(.5)
else: raise RuntimeError('Preview start timeout')
