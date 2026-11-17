"""Attach the exact immutable BC anchor to every live native policy export."""
from copy import deepcopy
import json
import os
from pathlib import Path
from native_eval.legwheel import policy_template as original_template
from .contract import ANCHOR_RATIO


def policy_template(metadata,ppo_config):
    result=original_template(metadata,ppo_config)
    descriptor=json.loads(Path(os.environ['TENNY_ALIGNED_ANCHOR_DESCRIPTOR']).read_text())
    anchor=deepcopy(descriptor['policy'])
    if 'leg_anchor' in anchor:raise ValueError('Anchor must be the original BC actor')
    anchor['output_activation']='identity'
    result.update(leg_anchor=anchor,leg_anchor_ratio=ANCHOR_RATIO)
    return result
