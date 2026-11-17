"""Run six frozen mature-policy fine tunes in separate GPU processes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(inputs, output):
    inputs,output=Path(inputs).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new matrix directory; retained runs are never restarted')
    output.mkdir(parents=True,exist_ok=True)
    freeze_path=inputs/'runtime-freeze.json'
    freeze_bytes=freeze_path.read_bytes()
    freeze=json.loads(freeze_bytes)
    completed=[]
    start=time.monotonic()
    for seed,order in [(0,['high_lr','low_lr']),(1,['low_lr','high_lr']),(2,['high_lr','low_lr'])]:
        for condition in order:
            if freeze_path.read_bytes()!=freeze_bytes:
                raise ValueError('Runtime freeze contract changed')
            for name,digest in freeze['files'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Frozen input changed: '+name)
            manifest=inputs/(condition+'.json')
            status=dict(phase='training',current_condition=condition,current_seed=seed,completed=completed,
                        runtime_freeze_sha256=hashlib.sha256(freeze_bytes).hexdigest(),
                        elapsed_seconds=time.monotonic()-start)
            write(output/'status.json',status)
            run_dir=output/f'seed-{seed}'/condition
            log=output/f'seed-{seed}-{condition}.log'
            print(json.dumps(dict(event='start',condition=condition,seed=seed,completed=len(completed))),flush=True)
            with log.open('w',encoding='utf-8') as stream:
                process=subprocess.run([sys.executable,'-B','-m','experiments.mature_balance_finetune',
                                        '--manifest',str(manifest),'--seed',str(seed),'--output-dir',str(run_dir)],
                                       stdout=stream,stderr=subprocess.STDOUT,timeout=720,check=False)
            if process.returncode!=0:
                write(output/'status.json',dict(status,phase='failed',error=f'Failed {condition}/{seed}; see {log}'))
                raise RuntimeError('Training failed; original error retained in '+str(log))
            result=json.loads((run_dir/'status.json').read_text())
            if result['phase']!='completed' or result['steps']!=2097152:
                raise RuntimeError('Incomplete run')
            completed.append(dict(result,condition=condition,seed=seed,reused=False))
            print(json.dumps(dict(event='completed',condition=condition,seed=seed,seconds=result['elapsed_seconds'])),flush=True)
    result=dict(phase='completed',completed=completed,total_steps=sum(r['steps'] for r in completed),
                runtime_freeze_sha256=hashlib.sha256(freeze_bytes).hexdigest(),
                elapsed_seconds=time.monotonic()-start)
    write(output/'status.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.inputs,args.output)
