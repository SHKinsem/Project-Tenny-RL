"""Shared force request coordinates; execution capacity is the only arm difference."""
from copy import deepcopy
import math
from experiments.torque_limit_ablation.contract import actuator_contract

ANCHOR_RATIO=.5*(1.-2.**-20)


def aligned_mean_scalar(current,anchor):
    return math.atanh(ANCHOR_RATIO*math.tanh(anchor))+(current-anchor)


def aligned_contract(limit,jmax):
    result=actuator_contract(limit,jmax)
    result.update(policy_leg_force_scales=actuator_contract(2.,jmax)['force_scales_n'],
        anchor_ratio=ANCHOR_RATIO,policy_coordinate='Common Fs2 request for both arms, then per-arm Fs capacity and motor clipping.')
    return result


def aligned_policy(base):
    result=deepcopy(base)
    if result.get('leg_anchor') or result.get('output_activation')!='tanh':
        raise ValueError('Need the original unsupplemented BC policy')
    anchor=deepcopy(base);anchor['output_activation']='identity'
    result.update(leg_anchor=anchor,leg_anchor_ratio=ANCHOR_RATIO)
    return result


def aligned_descriptor(base,limit,xml_path):
    result=deepcopy(base)
    contract=aligned_contract(limit,result['task']['config']['jmax'])
    result['xml_path']=str(xml_path)
    result['policy']=aligned_policy(base['policy'])
    cfg=result['task']['config']
    cfg.update(torque_scales=contract['torque_limits_nm'],
        torque_normalization_scales=contract['torque_normalization_scales'],
        force_scales=contract['force_scales_n'],policy_leg_force_scales=contract['policy_leg_force_scales'])
    result['metadata'].update(experimental_only=True,experiment_kind='aligned_torque_limit_ablation_v1',actuator_contract=contract)
    return result
