"""Six fresh paired runs over frozen common-policy inputs, in isolated processes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(inputs,output):
    inputs,output=Path(inputs).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Preserve previous or failed matrix output')
    output.mkdir(parents=True,exist_ok=True)
    manifests={limit:inputs/f'manifest-limit{limit}.json' for limit in (1,2)}
    values={limit:json.loads(path.read_text()) for limit,path in manifests.items()}
    receipt=json.loads((inputs/'runtime-freeze.json').read_text())
    receipt.update({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in manifests.values()})
    write(output/'frozen-inputs.json',receipt)
    completed=[];started=time.monotonic()
    for seed in (0,1,2):
        for limit in ((1,2) if seed!=1 else (2,1)):
            for path,expected in receipt.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Frozen runtime/input changed: '+path)
            condition=f'limit{limit}';target=output/f'seed-{seed}'/condition
            state=dict(phase='training',current_seed=seed,current_condition=condition,completed=completed,
                elapsed_seconds=time.monotonic()-started)
            write(output/'status.json',state)
            print(json.dumps(dict(event='start',seed=seed,condition=condition,completed=len(completed))),flush=True)
            command=[sys.executable,'-B','-m','experiments.reward_ablation','--manifest',str(manifests[limit]),
                '--arm','baseline','--seed',str(seed),'--output-dir',str(target)]
            log=output/f'seed-{seed}-{condition}.log'
            try:
                with log.open('w') as stream:
                    process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=600,check=False)
                if process.returncode:raise RuntimeError('Training failed; inspect '+str(log))
                status=json.loads((target/'status.json').read_text())
                if status['phase']!='completed' or status['steps']!=values[limit]['steps_per_run']:
                    raise RuntimeError('Run did not complete the exact budget')
                completed.append(dict(status,condition=condition,reused=False))
                print(json.dumps(dict(event='completed',seed=seed,condition=condition,
                    steps=status['steps'],elapsed_seconds=status['elapsed_seconds'])),flush=True)
            except (RuntimeError,subprocess.TimeoutExpired) as error:
                write(output/'status.json',dict(state,phase='failed',error=str(error),elapsed_seconds=time.monotonic()-started))
                raise
    result=dict(phase='completed',completed=completed,total_runs=6,
        total_environment_steps=sum(item['steps'] for item in completed),elapsed_seconds=time.monotonic()-started)
    write(output/'status.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args();run(options.inputs,options.output)
