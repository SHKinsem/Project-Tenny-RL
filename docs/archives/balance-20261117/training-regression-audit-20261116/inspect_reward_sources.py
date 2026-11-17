"""Read-only source archive/manifest inspection; no policy or simulator imports."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

repo=Path(__file__).resolve().parents[3]
experiment=repo/'tasks/results/aligned-limit-ablation-20261116'
archive=experiment/'comparison/actual-experiment-source.tar.gz'
parser=argparse.ArgumentParser()
parser.add_argument('--list',action='store_true')
parser.add_argument('--member')
parser.add_argument('--start',type=int,default=1)
parser.add_argument('--end',type=int,default=10000)
parser.add_argument('--manifest',action='store_true')
options=parser.parse_args()
if options.list:
    with tarfile.open(archive) as stream:
        print(json.dumps([dict(name=m.name,size=m.size,sha256=hashlib.sha256(stream.extractfile(m).read()).hexdigest())
            for m in stream.getmembers() if m.isfile()],indent=2))
if options.member:
    with tarfile.open(archive) as stream:
        lines=stream.extractfile(options.member).read().decode().splitlines()
    for number,line in enumerate(lines,1):
        if options.start<=number<=options.end:print(f'{number}: {line}')
if options.manifest:
    manifest=json.loads((experiment/'inputs/manifest-limit1.json').read_text())
    print(json.dumps(dict(ppo=manifest['ppo'],reward_config=manifest['env_config']['reward_config'],
        cmd_config=manifest['env_config']['cmd_config'],policy_metadata=manifest['policy_metadata'],
        initializer={k:v for k,v in manifest.items() if k.startswith('initial_') and k!='initial_states'},
        protected_inputs=manifest['protected_inputs']),indent=2))
