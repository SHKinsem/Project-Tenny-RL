"""Strict JSON input and non-overwriting result files."""
import json
import math
from pathlib import Path


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON field: '+key)
        result[key]=value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=_object,
                      parse_float=_float,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON value: '+value)))


def _float(text):
    value=float(text)
    if not math.isfinite(value):raise ValueError('Nonfinite JSON number: '+text)
    return value


def save_json(path,value):
    payload=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(payload)
