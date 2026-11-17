"""Same trainable BC parameters, frozen anchor leg means in common force units."""
import os
from .contract import ANCHOR_RATIO


def transform_logits(current,anchor,xp):
    if current.shape[-1] not in (4,8) or current.shape!=anchor.shape:
        raise ValueError('Need matching original four/eight mean/scale logits')
    legs=xp.arctanh(ANCHOR_RATIO*xp.tanh(anchor[...,:2]))+(current[...,:2]-anchor[...,:2])
    return xp.concatenate((legs,current[...,2:]),axis=-1)


def make_ppo_networks(*args,anchor_checkpoint=None,**kwargs):
    from dataclasses import replace
    import jax
    import jax.numpy as jp
    from legwheel.deployment.extract import load
    from force_teacher.networks import make_ppo_networks as original_factory
    path=anchor_checkpoint or os.environ['TENNY_ALIGNED_ANCHOR_CHECKPOINT']
    anchor_params=jax.tree.map(jp.asarray,load(path)[1])
    base=original_factory(*args,**kwargs)
    original_apply=base.policy_network.apply
    def apply(normalizer,params,observation):
        current=original_apply(normalizer,params,observation)
        anchor=original_apply(normalizer,anchor_params,observation)
        return transform_logits(current,anchor,jp)
    return base.replace(policy_network=replace(base.policy_network,apply=apply))
