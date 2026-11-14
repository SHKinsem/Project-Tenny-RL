"""Isolated HTTP preview acceptance with an actor publisher, never runs PPO."""
import os, sys, json, subprocess, tempfile, time, signal, urllib.request
from pathlib import Path
BASE=(Path.home() / "projects/tenny-torque200")
WIN=(Path(os.environ.get("TENNY_ROOT") or Path(__file__).resolve().parents[3]) / "experiments/torque200")
PY=str(Path.home() / "projects/tenny-reference/.venv-gpu/bin/python")
URL='http://127.0.0.1:8771'
sys.path.insert(0,str(BASE/'src'))
from training_control import TrainingController, _write
real=TrainingController(BASE/'artifacts',(Path.home() / "projects/tenny-reference"),BASE)
model=real.latest('original')
assert model
PUBLISHER = '''import sys,time,json
from pathlib import Path
sys.path.insert(0, {source!r})
from legwheel.deployment.extract import load
from preview_sync import SnapshotPublisher
run=Path(sys.argv[sys.argv.index('--run-dir')+1])
params=load({checkpoint!r})
publisher=SnapshotPublisher(run,'original')
(run/'status.json').write_text(json.dumps(dict(mode='original',phase='training',preview_protocol=1)))
steps=100
while True:
    if not (run/'hold').exists(): publisher.service(steps,params)
    steps+=100
    time.sleep(.1)
'''
def state():
    return json.load(urllib.request.urlopen(URL+'/state',timeout=2))
def control(action):
    req=urllib.request.Request(URL+'/control',json.dumps(dict(action=action)).encode(),{'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=2))
def until(predicate,seconds=45):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        try:
            result=predicate()
            if result:return result
        except OSError: pass
        time.sleep(.2)
    raise AssertionError('HTTP acceptance timed out')
with tempfile.TemporaryDirectory(prefix='preview-http-') as directory:
    tmp=Path(directory)
    run=tmp/'artifacts/console/original/test-run'
    run.mkdir(parents=True)
    script=tmp/'console_train.py'
    script.write_text(PUBLISHER.format(source=str(BASE/'src'),checkpoint=model['checkpoint']))
    env=os.environ.copy()
    env.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(BASE/'jax-cache'))
    env.pop('MUJOCO_GL',None)
    publisher=web=None
    try:
        with (tmp/'publisher.log').open('wb') as log:
            publisher=subprocess.Popen([PY,str(script),'--run-dir',str(run)],stdout=log,stderr=subprocess.STDOUT,env=env)
        until(lambda:(run/'status.json').is_file())
        _write(tmp/'artifacts/console/active.json',dict(mode='original',run_dir=str(run),pid=publisher.pid,script=str(script)))
        with (tmp/'web.log').open('wb') as log:
            web=subprocess.Popen([PY,str(BASE/'src/web_preview.py'),'--artifacts',str(tmp/'artifacts'),'--port','8771','--mode','original'],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=BASE/'src')
        until(lambda:state().get('sync_status')=='idle')
        control('resume')
        first=until(lambda:(s if (s:=state()).get('sync_status')=='ready' else None))
        assert first['preview_kind']=='live' and first['preview_training_steps']>=100
        identifier=first['preview_request_id']
        with urllib.request.urlopen(URL+'/stream.mjpg',timeout=15) as stream:
            chunk=stream.read(2048)
            assert b'Content-Type: image/jpeg' in chunk and b'\xff\xd8' in chunk
        snapshot=(run/'preview-policy.npz').read_bytes()
        control('resume') # duplicate open must not publish again
        time.sleep(.5)
        assert state()['preview_request_id']==identifier
        assert (run/'preview-policy.npz').read_bytes()==snapshot
        control('suspend')
        time.sleep(.2)
        assert not (run/'preview.request.json').exists()
        assert state()['suspended']
        frames=state()['frames']
        time.sleep(.3)
        assert state()['frames']==frames
        control('resume')
        second=until(lambda:(s if (s:=state()).get('sync_status')=='ready' and s['preview_request_id']!=identifier else None))
        assert second['preview_training_steps']>first['preview_training_steps']
        control('suspend')
        (run/'hold').touch()
        control('resume')
        assert state()['sync_status']=='pending'
        control('suspend')
        (run/'hold').unlink()
        time.sleep(.4)
        assert state()['suspended'] and state()['sync_status']=='idle'
        result=dict(first_steps=first['preview_training_steps'],second_steps=second['preview_training_steps'],
                    first_id=identifier,second_id=second['preview_request_id'],
                    jpeg_received=True,duplicate_open_unchanged=True,closed_frames_stopped=True,cancelled_open_stays_closed=True,
                    validation_scope='CPU actor publisher with real saved weights + real web simulator; no PPO training')
        (WIN/'results/preview-on-open/http-check.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    except Exception:
        for name in ('web.log','publisher.log'):
            if (tmp/name).exists(): print(name,(tmp/name).read_text(errors='replace')[-4000:])
        raise
    finally:
        for process in (web,publisher):
            if process and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
