"""Three fresh weak-wheel runs against explicitly identified reused controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def run(inputs,output):
    inputs,output=Path(inputs).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh matrix directory')
    output.mkdir(parents=True,exist_ok=True)
    freeze_path=inputs/'runtime-freeze.json';freeze_bytes=freeze_path.read_bytes();freeze=json.loads(freeze_bytes)
    start=time.monotonic();completed=[]
    for seed in [0,1,2]:
        if freeze_path.read_bytes()!=freeze_bytes:raise ValueError('Frozen contract changed')
        for name,digest in freeze['files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen input changed: '+name)
        state=dict(phase='training',current_seed=seed,completed=completed,elapsed_seconds=time.monotonic()-start,
                   runtime_freeze_sha256=hashlib.sha256(freeze_bytes).hexdigest())
        write(output/'status.json',state)
        print(json.dumps(dict(event='start',seed=seed,completed=len(completed))),flush=True)
        path=output/f'seed-{seed}'/'weak_wheel';log=output/f'seed-{seed}.log'
        try:
            with log.open('w') as stream:
                process=subprocess.run([sys.executable,'-B','-m','experiments.wheel_hold_ablation','--manifest',str(inputs/'manifest.json'),
                                        '--seed',str(seed),'--output-dir',str(path)],stdout=stream,stderr=subprocess.STDOUT,timeout=720)
            if process.returncode:raise RuntimeError('See '+str(log))
            result=json.loads((path/'status.json').read_text())
            if result['phase']!='completed' or result['steps']!=2097152:raise RuntimeError('Incomplete fixed budget')
            completed.append(dict(result,reused=False))
            print(json.dumps(dict(event='completed',seed=seed,seconds=result['elapsed_seconds'])),flush=True)
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            write(output/'status.json',dict(state,phase='failed',error=str(exc),elapsed_seconds=time.monotonic()-start));raise
    result=dict(phase='completed',completed=completed,total_new_steps=sum(row['steps'] for row in completed),
                elapsed_seconds=time.monotonic()-start,runtime_freeze_sha256=hashlib.sha256(freeze_bytes).hexdigest())
    write(output/'status.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.inputs,a.output)
