"""Explicit execution capability, with original reward/observation units fixed."""
from copy import deepcopy
import math

REFERENCE_SCALES = (1., 1., .0295, .0295)


def actuator_contract(leg_limit_nm, jmax):
    if isinstance(leg_limit_nm, bool) or leg_limit_nm not in (1., 2.):
        raise ValueError('This comparison supports 1 or 2 N m')
    if len(jmax) != 2 or any(not math.isfinite(v) or v <= 0 for v in jmax):
        raise ValueError('Need the two positive frozen Jacobian bounds')
    return dict(leg_limit_nm=float(leg_limit_nm),
        torque_limits_nm=[float(leg_limit_nm)]*2+list(REFERENCE_SCALES[2:]),
        torque_normalization_scales=list(REFERENCE_SCALES),
        force_scales_n=[.9*float(leg_limit_nm)/v for v in jmax],
        formula='Fs = 0.9 * execution_limit / frozen_Jmax',
        normalization='Original physical torque units for reward, rate history, and previous-torque observation.')


def experimental_descriptor(base, leg_limit_nm, xml_path):
    result=deepcopy(base)
    cfg=result['task']['config']
    if cfg['action_mode']!='virtual_leg_force' or cfg['torque_scales']!=list(REFERENCE_SCALES):
        raise ValueError('Need the unchanged baseline virtual-leg-force descriptor')
    contract=actuator_contract(leg_limit_nm,cfg['jmax'])
    result['xml_path']=str(xml_path)
    cfg['torque_scales']=contract['torque_limits_nm']
    cfg['torque_normalization_scales']=contract['torque_normalization_scales']
    cfg['force_scales']=contract['force_scales_n']
    result['metadata'].update(experimental_only=True,experiment_kind='torque_limit_ablation_v1',
        actuator_contract=contract)
    return result


def patch_frozen_trainer(source):
    replacements=[
        ('from .legwheel_env import RewardExperimentEnv',
         'from experiments.torque_limit_ablation.env import TorqueLimitEnv as RewardExperimentEnv'),
        ("root_body=descriptor['task']['config']['root_body'])",
         "root_body=descriptor['task']['config']['root_body'], leg_limit_nm=manifest['leg_limit_nm'])"),
        ("reward=manifest['arms'][arm], run_evaluation_reward=",
         "reward=manifest['arms'][arm], actuator_contract=manifest['actuator_contract'], run_evaluation_reward=")]
    for old,new in replacements:
        if source.count(old)!=1:
            raise ValueError('Unknown frozen training source: '+old)
        source=source.replace(old,new)
    return source
