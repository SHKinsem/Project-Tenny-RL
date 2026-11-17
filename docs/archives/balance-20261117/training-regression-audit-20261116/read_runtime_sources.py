"""Read/hash specific deployed or installed source files without importing them."""
import argparse
import hashlib
import json
from pathlib import Path

repo=Path(__file__).resolve().parents[3]
manifest=json.loads((repo/'tasks/results/aligned-limit-ablation-20261116/inputs/manifest-limit1.json').read_text())
parser=argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--start',type=int,default=1)
parser.add_argument('--end',type=int,default=10000)
parser.add_argument('--json',action='store_true')
parser.add_argument('--keys',action='store_true')
parser.add_argument('--key',action='append')
parser.add_argument('--find',action='append')
options=parser.parse_args()
path=Path(options.path)
data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
expected=manifest['protected_inputs']['files'].get(str(path))
print(json.dumps(dict(path=str(path),sha256=digest,expected_frozen_sha256=expected,
    matches_frozen=None if expected is None else digest==expected)))
if expected is not None:assert digest==expected,'Protected source differs from the formal input'
if options.json:
    value=json.loads(data)
    if options.keys:print(json.dumps(list(value)))
    if options.key:
        def selected(key):
            found=value
            for part in key.split('.'):
                if not isinstance(found,dict):return None
                found=found.get(part)
            return found
        print(json.dumps({key:selected(key) for key in options.key},indent=2))
    elif not options.keys:print(json.dumps(value,indent=2))
else:
    for number,line in enumerate(data.decode().splitlines(),1):
        if options.start<=number<=options.end and (not options.find or any(word in line for word in options.find)):
            print(f'{number}: {line}')
